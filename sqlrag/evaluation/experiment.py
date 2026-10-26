"""Controlled baseline-vs-RAG experiment.

Both conditions use the same SQL items, the same generator model and sampling
settings (temperature 0, fixed seed, same context window), the same judge and
rubric, and the same machine. The only intended difference is retrieval
augmentation: the RAG prompt (prompts/v3_rag.md) is the v2 prompt verbatim
plus a section on using sources, and its user message adds the retrieved
passages.

Phases (each resumable; finished rows are skipped on restart):
  generate  for every item, produce the baseline and the RAG explanation.
            The order of the two conditions alternates per item so warm-up
            and caching effects do not favour one condition.
  judge     grade every generated explanation with a judge configuration
            (model + mode). Several judges can be run on the same generations.
  summarize compute metrics, paired statistics and reports (compare.py).

Output: results/rag/<exp_id>/
  metadata.json                 dataset/prompt hashes, models, settings, environment
  generations.jsonl             one row per (item, condition)
  judgments__<judge_tag>.jsonl  one row per (item, condition) per judge
  summary__<judge_tag>.json     metrics (written by compare.py)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import pathlib
import platform
import subprocess
import sys
import time

from tqdm import tqdm

from ..config import ROOT, Settings, get_settings
from ..explain import Explainer
from ..judge import judge_tag, make_judge
from ..llm import make_client
from ..prompting import load_prompt
from ..retrieval import RetrievedChunk, Retriever
from ..schemas import Explanation
from ..sql_analysis import analyze_sql

logger = logging.getLogger(__name__)
CONDITIONS = ("baseline", "rag")


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _pkg_versions() -> dict:
    out = {}
    for name in ("ollama", "chromadb", "sentence-transformers", "torch", "sqlglot", "pydantic", "fastapi"):
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def load_items(split: str, data_dir: pathlib.Path) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping corrupt line in %s", path)
    return rows


def _compact_retrieval(res: dict | None) -> dict | None:
    if not res:
        return None
    return {
        "chunks": [
            {k: c.get(k) for k in ("source_label", "chunk_id", "doc_id", "title", "section", "source_url", "dense_score",
                                   "bm25_score", "fused_score", "covers", "n_tokens", "text")}
            for c in res.get("chunks", [])
        ],
        "required_features": res.get("required_features"),
        "covered_features": res.get("covered_features"),
        "missing_features": res.get("missing_features"),
        "coverage": res.get("coverage"),
        "sufficient": res.get("sufficient"),
        "reasons": res.get("reasons"),
        "config": res.get("config"),
        "timings": res.get("timings"),
        "queries": res.get("queries"),
    }


class Experiment:
    def __init__(self, exp_dir: pathlib.Path, settings: Settings) -> None:
        self.dir = pathlib.Path(exp_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.gen_path = self.dir / "generations.jsonl"
        self.meta_path = self.dir / "metadata.json"

    # ------------------------------------------------------------------
    def write_metadata(self, split: str, items: list[dict], explainer: Explainer, extra: dict) -> dict:
        s = self.settings
        data_file = s.data_dir / f"{split}.jsonl"
        client = explainer.client
        meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
        kb = explainer.retriever.kb if explainer.retriever else None
        meta.update({
            "exp_id": self.dir.name,
            "created_at": meta.get("created_at") or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "research_question": "Does retrieval augmentation improve the correctness, completeness and reliability of "
                                 "LLM-generated SQL explanations compared with the prompt-only baseline?",
            "dataset": {"split": split, "file": str(data_file.relative_to(ROOT)) if data_file.is_relative_to(ROOT) else str(data_file),
                        "sha256": sha256_file(data_file), "n_items": len(items)},
            "conditions": {
                "baseline": {"prompt": s.baseline_prompt.name, "prompt_sha256_12": explainer.baseline_hash, "retrieval": False},
                "rag": {"prompt": s.rag_prompt.name, "prompt_sha256_12": explainer.rag_hash, "retrieval": True,
                        "grounding_policy": s.grounding_policy},
            },
            "generator": {"model": s.generator_model, "provider": s.llm_provider,
                          "identity": client.model_identity(s.generator_model) if hasattr(client, "model_identity") else None,
                          "options": s.generation_options()},
            "retrieval": {**s.retrieval_config(), **({"collection": kb.collection_name, "index_stats": kb.stats()} if kb else {})},
            "seed": s.seed,
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
                            "packages": _pkg_versions(), "git_commit": _git_commit()},
            **extra,
        })
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return meta

    # ------------------------------------------------------------------
    def generate(self, items: list[dict], explainer: Explainer, conditions: tuple[str, ...] = CONDITIONS) -> None:
        done = {(r["id"], r["condition"]) for r in read_jsonl(self.gen_path) if not r.get("error")}
        todo = [(i, it) for i, it in enumerate(items) if any((it["id"], c) not in done for c in conditions)]
        logger.info("generation: %d/%d items already complete", len(items) - len(todo), len(items))
        if not todo:
            return
        # Warm-up so model loading time is not charged to the first measured item.
        try:
            explainer.client.generate_json(self.settings.generator_model, "Reply with JSON.", "Say ok in ten words or more.",
                                           Explanation, self.settings.generation_options(), max_retries=1)
        except Exception as exc:
            logger.warning("warm-up call failed: %s", exc)
        with self.gen_path.open("a", encoding="utf-8") as fh, tqdm(total=len(todo), unit="item", desc="generate") as bar:
            for idx, item in todo:
                order = conditions if idx % 2 == 0 else tuple(reversed(conditions))
                for cond in order:
                    if (item["id"], cond) in done:
                        continue
                    row = self._generate_one(item, cond, explainer, order.index(cond))
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                bar.update(1)

    def _generate_one(self, item: dict, cond: str, explainer: Explainer, position: int) -> dict:
        base = {"id": item["id"], "condition": cond, "feature": item.get("feature"), "domain": item.get("domain"),
                "sql": item["sql"], "schema_sha256_12": hashlib.sha256(item["schema"].encode()).hexdigest()[:12],
                "position_in_pair": position, "should_abstain": item.get("should_abstain"),
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        try:
            r = explainer.explain(item["sql"], item["schema"], mode=cond)
            return {
                **base,
                "explanation": r.get("explanation"),
                "explanation_clean": r.get("explanation_clean"),
                "draft_explanation": r.get("draft_explanation"),
                "citations": r.get("citations"),
                "citation_report": r.get("citation_report"),
                "grounding_status": r.get("grounding_status"),
                "abstained": r.get("abstained", False),
                "abstention_stage": r.get("abstention_stage"),
                "abstention_reason": r.get("abstention_reason"),
                "model_evidence_sufficient": r.get("model_evidence_sufficient"),
                "retrieval": _compact_retrieval(r.get("retrieval")),
                "metrics": r.get("metrics"),
                "model": r.get("model"),
                "prompt_file": r.get("prompt_file"),
                "prompt_hash": r.get("prompt_hash"),
                "error": None,
            }
        except Exception as exc:
            logger.error("generation failed for %s/%s: %s", item["id"], cond, exc)
            return {**base, "explanation": None, "explanation_clean": None, "error": str(exc)}

    # ------------------------------------------------------------------
    def judge(self, items: list[dict], judge, *, evidence_mode: str = "none") -> pathlib.Path:
        tag = judge_tag(judge.model, judge.mode) + ("__evidence" if evidence_mode == "retrieved" else "")
        out_path = self.dir / f"judgments__{tag}.jsonl"
        schemas = {it["id"]: it["schema"] for it in items}
        gens = [g for g in read_jsonl(self.gen_path) if not g.get("error")]
        latest: dict[tuple[str, str], dict] = {}
        for g in gens:
            latest[(g["id"], g["condition"])] = g
        rag_evidence = {k[0]: g.get("retrieval") for k, g in latest.items() if k[1] == "rag"}
        done = {(r["id"], r["condition"]) for r in read_jsonl(out_path) if not r.get("error")}
        todo = [g for k, g in latest.items() if k not in done and g["id"] in schemas]
        logger.info("judging with %s: %d to do, %d done", tag, len(todo), len(done))
        with out_path.open("a", encoding="utf-8") as fh:
            for g in tqdm(todo, unit="expl", desc=f"judge {tag}"):
                row = {"id": g["id"], "condition": g["condition"], "judge_tag": tag,
                       "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
                text = g.get("explanation_clean")
                if not text:
                    row.update({"skipped": "abstained" if g.get("abstained") else "no_explanation", "score": None, "error": None})
                else:
                    evidence = None
                    if evidence_mode == "retrieved" and rag_evidence.get(g["id"]):
                        evidence = [RetrievedChunk(chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"], section=c["section"],
                                                   source_url=c.get("source_url", ""), license="", doc_type="", dialect="",
                                                   text=c.get("text", ""), n_tokens=c.get("n_tokens", 0), source_label=c["source_label"])
                                    for c in rag_evidence[g["id"]]["chunks"]]
                    try:
                        row.update(judge.grade(g["sql"], schemas[g["id"]], text, analysis=analyze_sql(g["sql"]), evidence=evidence))
                        row["error"] = None
                    except Exception as exc:
                        logger.error("judge failed for %s/%s: %s", g["id"], g["condition"], exc)
                        row.update({"score": None, "error": str(exc)})
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
        return out_path


def export_legacy(exp_dir: pathlib.Path, judgments_path: pathlib.Path, split: str, results_dir: pathlib.Path) -> list[pathlib.Path]:
    """Write per-condition files in the original results format so the legacy
    dashboard, harness/analyze.py and harness/export_pdf.py can read them."""
    gens = {(g["id"], g["condition"]): g for g in read_jsonl(exp_dir / "generations.jsonl") if not g.get("error")}
    judg = {(j["id"], j["condition"]): j for j in read_jsonl(judgments_path) if j.get("score") is not None}
    tag = judgments_path.stem.split("__", 1)[1].replace("__", "-")
    paths = []
    for cond in CONDITIONS:
        # <exp>-<condition>-<judge>_<split>.jsonl: one file per judge, readable by the legacy dashboard.
        out = results_dir / f"{exp_dir.name}-{cond}-{tag}_{split}.jsonl"
        rows = []
        for (iid, c), g in gens.items():
            if c != cond or (iid, c) not in judg:
                continue
            j = judg[(iid, c)]
            m = g.get("metrics") or {}
            rows.append({
                "id": iid, "input": g["sql"], "output": g.get("explanation_clean"), "score": j["score"], "reason": j.get("reason"),
                "sys_cost": {"latency": m.get("total_s", 0), "usd": 0.0, "in_tok": m.get("in_tok", 0), "out_tok": m.get("out_tok", 0)},
                "judge_cost": {k: (j.get("cost") or {}).get(k, 0) for k in ("latency", "usd", "in_tok", "out_tok")},
                "error": None,
            })
        if rows:
            out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            paths.append(out)
    return paths


def main(args) -> int:
    settings = get_settings(
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        grounding_policy=args.policy,
        embedding_backend=getattr(args, "embedding_backend", None),
        embedding_model=getattr(args, "embedding_model", None),
        chunk_size=getattr(args, "chunk_size", None),
    )
    items = load_items(args.split, settings.data_dir)
    if args.limit:
        items = items[: args.limit]
    exp = Experiment(settings.results_dir / "rag" / args.exp_id, settings)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    phases = {"generate", "judge", "summarize"} if args.phase == "all" else {args.phase}
    client = make_client(settings)

    if "generate" in phases:
        from ..index.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings)
        if kb.store.count() == 0:
            logger.info("index is empty; ingesting the default corpus first")
            kb.ingest_manifest()
        explainer = Explainer(settings, client, Retriever(kb, settings))
        exp.write_metadata(args.split, items, explainer, {"limit": args.limit, "conditions_run": list(conditions)})
        t0 = time.time()
        exp.generate(items, explainer, conditions)
        logger.info("generation phase finished in %.1f min", (time.time() - t0) / 60)

    judgments_path = None
    if "judge" in phases:
        judge = make_judge(settings, client, settings.judge_model, args.judge_mode)
        judgments_path = exp.judge(items, judge, evidence_mode=args.judge_evidence)
        meta = json.loads(exp.meta_path.read_text(encoding="utf-8")) if exp.meta_path.exists() else {}
        judges = meta.setdefault("judges", {})
        tag = judgments_path.stem.split("__", 1)[1]
        _, jhash = load_prompt(settings.judge_prompt if args.judge_mode == "rubric" else ROOT / "prompts" / "judge.md")
        # num_gpu only changes speed (layer placement), not sampling.
        judges[tag] = {"model": settings.judge_model, "mode": args.judge_mode, "evidence": args.judge_evidence,
                       "prompt_sha256_12": jhash, "options": settings.generation_options(),
                       "identity": client.model_identity(settings.judge_model) if hasattr(client, "model_identity") else None}
        exp.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        export_legacy(exp.dir, judgments_path, args.split, settings.results_dir)

    if "summarize" in phases:
        from .compare import summarize_experiment

        for path in sorted(exp.dir.glob("judgments__*.jsonl")):
            summary = summarize_experiment(exp.dir, path)
            logger.info("summary written for %s", path.name)
            if not args.no_pdf:
                try:
                    from .report_pdf import write_experiment_pdf

                    pdf = write_experiment_pdf(exp.dir, summary, settings.reports_dir)
                    logger.info("PDF report: %s", pdf)
                except Exception as exc:
                    logger.warning("PDF report failed: %s", exc)
    return 0
