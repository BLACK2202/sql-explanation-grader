"""
harness/analyze.py — Break down evaluation results by SQL feature, domain, and error type.

Usage:
    python harness/analyze.py results/v2_dev.jsonl
    python harness/analyze.py results/v2_dev.jsonl results/v2_test.jsonl   # compare two files
"""

import argparse
import json
import pathlib
import sys

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

parser = argparse.ArgumentParser(description="Analyze grader results by feature and domain.")
parser.add_argument("results", nargs="+", type=pathlib.Path, help="One or more results JSONL files.")
parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"),
                    help="Directory containing the source JSONL splits.")
parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
args = parser.parse_args()

if args.no_color:
    RESET = BOLD = GREEN = RED = YELLOW = CYAN = DIM = ""


def color_pct(pct: float) -> str:
    c = GREEN if pct >= 60 else (YELLOW if pct >= 40 else RED)
    return f"{c}{pct:5.1f}%{RESET}"


def load_results(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_metadata(data_dir: pathlib.Path) -> dict[str, dict]:
    """Build a map of item_id → {feature, domain} from all split files."""
    meta = {}
    for split_file in data_dir.glob("*.jsonl"):
        for line in split_file.open(encoding="utf-8"):
            try:
                item = json.loads(line)
                if "id" in item:
                    meta[item["id"]] = {
                        "feature": item.get("feature", "unknown"),
                        "domain":  item.get("domain",  "unknown"),
                    }
            except json.JSONDecodeError:
                pass
    return meta


def analyze(rows: list[dict], meta: dict[str, dict], label: str):
    scored = [r for r in rows if r.get("error") is None]
    errors = [r for r in rows if r.get("error") is not None]
    n_good = sum(r["score"] for r in scored)
    pct    = 100 * n_good / len(scored) if scored else 0

    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {label}{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")
    print(f"  Total items : {len(rows)}")
    print(f"  Scored      : {len(scored)}   ({len(errors)} errors)")
    print(f"  Good        : {n_good}  {color_pct(pct)}")

    # ── By feature ────────────────────────────────────────────
    print(f"\n{BOLD}By SQL Feature:{RESET}")
    feat_stats: dict[str, list] = {}
    for r in scored:
        feat = meta.get(r["id"], {}).get("feature", "unknown")
        feat_stats.setdefault(feat, []).append(r["score"])

    if feat_stats:
        col = 42
        header = f"  {'Feature':<{col}} {'Good':>5}  {'Total':>5}  {'Acc':>7}"
        print(f"{DIM}{header}{RESET}")
        print(f"  {'-'*col}  {'─'*5}  {'─'*5}  {'─'*7}")
        for feat, scores in sorted(feat_stats.items(), key=lambda x: -sum(x[1])/len(x[1])):
            g = sum(scores); t = len(scores); p = 100*g/t
            print(f"  {feat:<{col}} {g:>5}  {t:>5}  {color_pct(p)}")
    else:
        print("  (no feature metadata found — run from the project root)")

    # ── By domain ─────────────────────────────────────────────
    print(f"\n{BOLD}By Domain:{RESET}")
    dom_stats: dict[str, list] = {}
    for r in scored:
        dom = meta.get(r["id"], {}).get("domain", "unknown")
        dom_stats.setdefault(dom, []).append(r["score"])

    if dom_stats:
        col = 16
        header = f"  {'Domain':<{col}} {'Good':>5}  {'Total':>5}  {'Acc':>7}"
        print(f"{DIM}{header}{RESET}")
        print(f"  {'-'*col}  {'─'*5}  {'─'*5}  {'─'*7}")
        for dom, scores in sorted(dom_stats.items(), key=lambda x: -sum(x[1])/len(x[1])):
            g = sum(scores); t = len(scores); p = 100*g/t
            print(f"  {dom:<{col}} {g:>5}  {t:>5}  {color_pct(p)}")

    # ── Latency ───────────────────────────────────────────────
    lats = [
        r["sys_cost"]["latency"] + r["judge_cost"]["latency"]
        for r in scored if r.get("sys_cost") and r.get("judge_cost")
    ]
    if lats:
        print(f"\n{BOLD}Latency (generator + judge):{RESET}")
        print(f"  Avg : {sum(lats)/len(lats):.1f}s")
        print(f"  Min : {min(lats):.1f}s")
        print(f"  Max : {max(lats):.1f}s")

    # ── Errors ────────────────────────────────────────────────
    if errors:
        print(f"\n{BOLD}{YELLOW}Errors ({len(errors)}):{RESET}")
        for r in errors[:10]:
            print(f"  {RED}{r['id']}{RESET}: {r.get('error','?')[:100]}")
        if len(errors) > 10:
            print(f"  … and {len(errors)-10} more")

    # ── Sample bad explanations ────────────────────────────────
    bads = [r for r in scored if r["score"] == 0 and r.get("reason")][:3]
    if bads:
        print(f"\n{BOLD}Sample BAD items:{RESET}")
        for r in bads:
            feat = meta.get(r["id"], {}).get("feature", "?")
            sql_preview = (r.get("input") or "")[:80].replace("\n", " ")
            print(f"\n  {DIM}[{r['id']} · {feat}]{RESET}")
            print(f"  SQL    : {sql_preview}")
            print(f"  Reason : {RED}{(r.get('reason') or '')[:120]}{RESET}")

    print()


# ── Main ──────────────────────────────────────────────────────
meta = load_metadata(args.data_dir)

all_file_rows = []
for path in args.results:
    if not path.exists():
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        continue
    rows = load_results(path)
    all_file_rows.append((path.stem, rows))
    analyze(rows, meta, label=path.stem)

# If multiple files given, print a comparison table
if len(all_file_rows) > 1:
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  Comparison{RESET}")
    print(f"{CYAN}{'═'*60}{RESET}")
    col = 30
    print(f"  {'Run':<{col}} {'Good':>5}  {'Total':>5}  {'Acc':>7}")
    print(f"  {'-'*col}  {'─'*5}  {'─'*5}  {'─'*7}")
    for label, rows in all_file_rows:
        scored = [r for r in rows if r.get("error") is None]
        g = sum(r["score"] for r in scored)
        t = len(scored)
        p = 100*g/t if t else 0
        print(f"  {label:<{col}} {g:>5}  {t:>5}  {color_pct(p)}")
    print()
