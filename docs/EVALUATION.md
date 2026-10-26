# Evaluation

**Research question.** Does Retrieval-Augmented Generation improve the correctness, completeness and reliability of LLM-generated SQL explanations compared with the original prompt-only approach?

**Short answer from our measurements.** Not with the small local generator we could run. With `llama3.2:3b`, the first RAG design (v3) produced *worse* explanations than the prompt-only baseline, because the model copied general documentation into its answers. A focused design (v4) removed most of that harm but did not beat the baseline. RAG did deliver what it can guarantee by construction: verifiable citations, evidence checks, and correct abstention on undocumented features. All numbers below come from runs in this repository; nothing is estimated.

| Result | Baseline (prompt only) | RAG |
|---|---|---|
| Dev, reference labels "good" (AI-assisted, 50 pairs) | 20/50 (40%) | v3: 10/50 (20%), −20 pts, McNemar p = 0.053 · v4: 16/50 (32%), −8 pts, p = 0.54 |
| **Test (held out), reference labels "good" (random 40 of 116, blind)** | 16/40 (40%) | v4: 11/40 (28%), −12.5 pts (95% CI −30 to +5), p = 0.27 |
| Dev, 7B judge approval (unverified) | 41/50 | v3: 30/50 |
| Citations pointing to a passage that was not provided | n/a | 0 (dev v3, dev v4, test) |
| Abstention probe (should abstain / should answer) | answers all 20 | 10/10 abstained, 10/10 answered |
| Raw SQL keywords in the text (test, lower is better) | 17/116 | 71/115 |
| Mean time per explanation (test) | 5.6 s | 10.2 s |
| Mean input tokens (test) | 573 | 1,434 |


## 1. What is measured, and how much each measurement can be trusted

| Measurement | Produced by | Trust level |
|---|---|---|
| Reference labels | Claude (an AI assistant) reading each explanation against the SQL and schema, mostly blind to the condition, with written rules (`data/labels/README.md`) | A careful second opinion, **not human ground truth**. No human labels exist yet. |
| Judge approval | An LLM judge (`qwen2.5:3b` or `qwen2.5:7b`) | Unverified; its agreement with the reference labels is reported in section 5. |
| Deterministic checks | Rules computed from the SQL structure (LIMIT value mentioned, sort direction words, aggregates named, invented numbers, SQL keyword leaks) | Reproducible but shallow: they check that operations are mentioned, not that the description is true. |
| Citation and grounding metrics | Deterministic validation of `[S#]` markers against the provided passages | Exact for validity; "lexical support" is a heuristic. |
| Retrieval metrics | Section-level relevance map (`data/retrieval_qrels.json`) | Rule-based; the map was written by an AI assistant from section headings and is not human-reviewed. |
| Abstention | Probe set with features documented or not documented by construction | Exact for this probe; small and clear-cut. |

## 2. Experimental setup

| | Baseline | RAG v3 (first design) | RAG v4 (focused) |
|---|---|---|---|
| System prompt | `prompts/v2.md` | `v2.md` verbatim + section on using sources (`v3_rag.md`) | `v2.md` verbatim + stricter source-usage rules (`v4_rag.md`) |
| Retrieved context | none | up to 6 passages, ≤ 1,800 tokens (mean 1,451) | one covering passage per SQL feature, ≤ 4 passages, ≤ 1,200 tokens (mean 486) |
| Generator | `llama3.2:3b` (Ollama), temperature 0, seed 42, 8,192-token context, ≤ 1,024 output tokens | same | same |
| Items | dev split, 50 queries (`data/dev.jsonl`, sha256 `65bb0817d4…`) | same | same |
| Judges | `qwen2.5:3b` rubric, `qwen2.5:3b` original prompt with and without schema, `qwen2.5:7b` rubric | same | same |

Controls: identical items, generator, sampling options and judges; the condition order alternates per item; the judge never sees which condition produced a text and never sees the retrieved documents (`[S#]` markers are stripped before judging). The baseline explanations were generated once and reused for the v4 comparison (identical prompt, model and options). Full metadata (dataset and prompt hashes, model digests, retrieval configuration, environment, git commit) is in each `results/rag/<experiment>/metadata.json`.

Hardware: Windows 11 laptop, NVIDIA RTX 2050 (4 GB), CPU embeddings. The 3B models ran fully on the GPU (`SQLRAG_NUM_GPU=99`); the 7B judge did not fit and ran about 58% on the CPU.

Why these models: the project's default generator `qwen2.5:7b` does not fit the 4 GB GPU, and the original report (`report.md`) used `llama3.2:3b` to generate and `qwen2.5:3b` to judge, so we kept that pair and added `qwen2.5:7b` as a stronger second judge.

## 3. Explanation quality

### 3.1 Reference labels (AI-assisted)

| Comparison (dev, 50 pairs) | Baseline good | RAG good | RAG − baseline (95% bootstrap CI) | Pairs: only baseline good / only RAG good | Exact McNemar p |
|---|---|---|---|---|---|
| Baseline vs RAG v3 | 20/50 (40%) | 10/50 (20%) | −20 pts (−38 to −2) | 16 / 6 | 0.053 |
| Baseline vs RAG v4 | 20/50 (40%) | 16/50 (32%) | −8 pts (−26 to +12) | 14 / 10 | 0.54 |

Labels: 100 blind labels for the first experiment; 50 labels for the v4 answers, which were **not** blind (all were known to be RAG answers). v4 was designed after reading the v3 errors on the same dev items, so its dev result is optimistic; see section 7 for the held-out check.

Why RAG v3 lost (from the label notes): the model pasted documentation statements that are false or irrelevant for the query at hand: an inner join described as a "cartesian product" or as equivalent to a CROSS JOIN, "compound SELECT" for a single query, wrong column counts ("books has 3 columns"), "OFFSET" or "WITH" mentioned where absent, and ROW_NUMBER/RANK tie behaviour confused with each other. 33 of 50 v3 answers contained raw SQL keywords, against 5 of 50 baseline answers.

Errors common to both conditions: invented relationships ("doctors who treated these patients", "students" counted where no student data is involved), "most recent" claims when the query sorts by another column, and omitted returned columns (for example the COUNT(*) value).

### 3.2 LLM-judge results

| Judge | Baseline approval | RAG v3 approval | RAG v4 approval |
|---|---|---|---|
| `qwen2.5:3b`, new rubric prompt with schema | 50/50 | 49/50 | 50/50 |
| `qwen2.5:3b`, original prompt without schema | 24/50 (22/50 on re-run) | 14/50 | 16/50 |
| `qwen2.5:3b`, original prompt with schema | 15/50 | 11/50 | 11/50 |
| `qwen2.5:7b`, new rubric prompt with schema | 41/50 | 30/50 | <!-- 7B-V4 --> |

The 7B judge, on paired items, found RAG v3 worse than the baseline: only-baseline-approved 15 vs only-RAG-approved 4 (exact McNemar p = 0.019), with lower mean scores on every dimension (groundedness −0.90, 95% CI −1.44 to −0.40; clarity −0.46; completeness −0.44; correctness −0.40, CI −0.84 to +0.02).

Judge reproducibility: re-judging the same 50 baseline texts with identical settings changed 0 verdicts; when only the GPU layer placement and output-length option differed (first run vs re-run of the original-prompt judge), 2 of 50 verdicts flipped. Differences of one or two items are therefore within judge noise.

These approvals should be read together with section 5: the 3B rubric judge approved almost everything and its agreement with the reference labels is at chance level.

### 3.3 Deterministic checks (no LLM)

| Check (dev) | Baseline | RAG v3 | RAG v4 |
|---|---|---|---|
| Mean coverage score | 0.960 | 0.980 | 0.973 |
| LIMIT value mentioned | 6/8 | 8/8 | 6/8 |
| Sort direction stated | 7/8 | 8/8 | 7/8 |
| COUNT named | 4/5 | 5/5 | 5/5 |
| All tables mentioned | 48/50 | 47/50 | 49/50 |
| Raw SQL keywords in the text (lower is better) | 5/50 | 33/50 | 30/50 |
| Numbers not present in the SQL (lower is better) | 5/50 | 6/50 | 5/50 |

The checks reward mentioning each operation, which RAG does slightly more often; they cannot see a false statement about the operation. This is why they disagree with the reference labels and why we report both.

## 4. Grounding and abstention

| RAG metric (dev) | v3 | v4 |
|---|---|---|
| Evidence judged sufficient before generation | 50/50 | 50/50 |
| Answers with at least one citation | 45/50 | 32/50 |
| Citation validity (cited source was really provided) | 1.000 | 1.000 |
| Answers citing a source that was not provided | 0 | 0 |
| Share of sentences carrying a citation | 0.45 | 0.31 |
| Lexical support of cited sentences (heuristic) | 0.81 | 0.69 |
| Grounding status | 45 grounded, 5 weakly grounded | 32 grounded, 18 weakly grounded |

The model never cited a passage it had not been given. A valid citation does not make the sentence true: many incorrect v3 sentences were faithfully cited, because the cited passage said something general that the model applied wrongly.

**Abstention probe** (`data/abstention_probe.jsonl`, 20 queries; RAG v3 configuration, policy `abstain`):

| | Should abstain (JSON or math functions, not in the default corpus) | Should answer (documented features) |
|---|---|---|
| Abstained | 10 | 0 |
| Answered | 0 | 10 |

Precision 1.0, recall 1.0. All 10 abstentions happened at the retrieval stage, before any LLM call (0 tokens, at most 0.3 s). The baseline answered all 20, including the 10 it had no documentation for. After ingesting the SQLite JSON page into a copy of the index (58 chunks, 8.9 s), all 5 JSON queries had sufficient evidence (`results/rag/json_ingestion_demo.json`). The probe is small and clear-cut by construction: it shows the mechanism works, not how it performs on subtle cases (for example a documented function used in an undocumented way).

## 5. How far can the judge be trusted?

Agreement with the AI-assisted reference labels on the same explanations (positive class = good):

| Judge configuration | n | Accuracy | Cohen's κ | Recall on bad explanations | Judge "good" rate (reference: 30%) |
|---|---|---|---|---|---|
| `qwen2.5:3b`, rubric + schema | 100 | 0.31 | 0.01 | 0.01 | 99% |
| `qwen2.5:3b`, original prompt, no schema | 100 | 0.64 | 0.20 | 0.69 | 38% |
| `qwen2.5:3b`, original prompt + schema | 100 | 0.60 | 0.01 | 0.74 | 26% |
| `qwen2.5:7b`, rubric + schema | 100 | 0.57 | 0.26 | 0.40 | 71% |

Findings:

* The schema-aware rubric judge with a 3B model is not usable: it approves nearly everything. The longer rubric prompt and the operation inventory appear to make a small judge more lenient, not more careful.
* The original short binary prompt is a better 3B judge, but κ = 0.20 is only slight agreement.
* **Adding the schema did not help a 3B judge.** With the same original prompt, giving it the schema (what `harness/run.py` now sends) made it stricter (26% approved instead of 38%) but its agreement fell from κ = 0.20 to 0.01. The schema is necessary in principle, since without it a judge cannot check column or table claims, but a 3B model does not use it well. This is why the fix is kept but its benefit is not claimed.
* The 7B judge is the most useful LLM judge we tested (κ = 0.26, fair agreement) but still lenient: it approved 71% where the reference labels found 30% good, and caught 40% of the bad explanations. It almost never rejected a good one (29 of 30 approved).
* Every judge that discriminates at all (3B original prompt, 7B rubric) ranks RAG v3 below the baseline, the same direction as the reference labels. The 3B rubric judge alone would have reported no difference.
* No human labels exist yet. The dashboard's blind labelling page writes `label_source: "human"` labels; adding even 50 of them would allow checking both the LLM judge and the AI-assisted labels.

## 6. Retrieval

166 queries (all splits), no LLM involved, relevance by section (`data/retrieval_qrels.json`). Context recall = share of a query's SQL features with a relevant passage among the passages actually given to the generator (selection with up to 6 passages).

| Chunk size | Query strategy | Mode | MRR | Recall@5 | Precision@5 | nDCG@5 | Context recall | ms / query |
|---|---|---|---|---|---|---|---|---|
| 400 | one query per feature | dense | 0.824 | 0.925 | 0.556 | 0.616 | 0.892 | 161 |
| 400 | one query per feature | BM25 | 0.642 | 0.836 | 0.568 | 0.560 | 0.960 | 5 |
| 400 | one query per feature | hybrid | 0.690 | 0.867 | 0.585 | 0.590 | 0.893 | 162 |
| 400 | single feature-list query | hybrid | 0.564 | 0.596 | 0.397 | 0.417 | 0.711 | 50 |
| 400 | raw SQL as the query | hybrid | 0.691 | 0.797 | 0.522 | 0.538 | 0.799 | 58 |
| 200 | one query per feature | hybrid | 0.640 | 0.878 | 0.604 | 0.569 | 0.948 | 163 |
| **300** | **one query per feature** | **hybrid** | 0.644 | 0.834 | 0.535 | 0.531 | **0.940** | 160 |
| 300 | one query per feature | BM25 | 0.658 | 0.842 | 0.597 | 0.576 | 0.962 | 5 |
| 500 | one query per feature | hybrid | 0.647 | 0.854 | 0.566 | 0.564 | 0.874 | 166 |
| 600 | one query per feature | hybrid | 0.615 | 0.830 | 0.490 | 0.502 | 0.899 | 160 |

Full table: `results/rag/retrieval_eval.json` (dense, BM25 and hybrid for every size).

Findings:

* **Decomposing the SQL into one query per feature matters most**: context recall 0.89 against 0.80 (raw SQL) and 0.71 (single list query) at the same chunk size.
* **Chunks above 400 tokens overflow the embedder.** At 500 and 600 tokens, 66 of 178 and 64 of 155 chunks exceeded bge-small's 512-token window and were truncated; dense quality dropped. 300 tokens kept every chunk inside the window (chosen setting).
* **Hybrid is not the best mode on every metric here.** BM25 alone gave the highest context recall (0.96) because SQL features have distinctive keywords; dense gave the best MRR. Hybrid sits in between. We kept hybrid as the default because it also serves free-text questions in the dashboard where exact keywords are absent, but this choice is not proven better by these numbers.
* **Results are sensitive to small settings.** Changing the overlap from 60 to 50 tokens moved hybrid context recall from 0.940 to 0.887. Differences of a few points between rows should not be over-interpreted.
* Retrieval takes about 0.2 s per query on the CPU (embedding the feature queries dominates); BM25 alone takes 5 ms.

## 7. Held-out check (test split)

Because v4 was designed after inspecting the dev errors, it was run once on the 116 test queries that had not been looked at (`results/rag/rag_v4_test`), with the same generator, settings and judges. A random sample of 40 test queries (seed 2026; 80 explanations) was labelled blind with the same decision rules.

| Test split | Baseline | RAG v4 |
|---|---|---|
| Reference labels "good" (40-item sample) | 16/40 (40%) | 11/40 (28%) |
| Paired: only baseline good / only RAG good / both good / both bad | 9 / 4 / 7 / 20 | |
| RAG − baseline (95% bootstrap CI), exact McNemar p | −12.5 pts (−30 to +5), p = 0.27 | |
| Generations (errors) | 116 (0) | 116 (0): 82 grounded, 33 weakly grounded, 1 abstained by the model |
| Evidence judged sufficient before generation | n/a | 116/116 |
| Answers with a citation / citing a passage not provided | n/a | 83/116 / 0 |
| Raw SQL keywords in the text | 17/116 | 71/115 |
| Mean time, input tokens, output tokens | 5.6 s, 573, 105 | 10.2 s, 1,434, 177 |

The held-out result agrees with dev: the focused RAG design is not better than the prompt-only baseline with this generator, and the point estimate is again slightly negative. The most frequent RAG-specific error on test was the claim that the joins "produce the same result as a CROSS JOIN", a sentence adapted from the SQLite join documentation that is false for joins with an `ON` condition. It caused 6 of the 29 RAG answers labelled bad and appeared in no baseline answer, often in explanations that were otherwise correct. Errors shared by both conditions were invented meaning ("members who have borrowed more than three books" for a query that counts members with the same name), missing returned columns, and "most recent" claims when the query sorts by another column.

Two test queries show why literal SQL semantics matter: `SELECT * FROM books WHERE author IN (SELECT author FROM members)` refers to the outer `books.author` because `members` has no `author` column, and `AVG(loan_date) > '2020-01-01'` compares a number with a text value. Neither condition explained these correctly.


## 8. Cost and latency

Per explanation on the laptop (dev, generation only; judge time excluded):

| | Baseline | RAG v3 | RAG v4 |
|---|---|---|---|
| Mean end-to-end time | 6.6 s | 17.5 s | 7.7 s |
| of which retrieval | 0 s | 1.5 s | 0.6 s |
| Mean input tokens | 571 | 2,647 | 1,469 |
| Mean output tokens | 104 | 266 | 191 |
| Mean explanation length | 83 words | 101 words | 84 words |
| Generation errors | 0 | 0 | 0 |

v3 latency includes periods when the machine was also running the test suite; treat the absolute values as indicative. Token counts are exact (reported by Ollama). Local inference has no monetary cost.

## 9. Historical results (before this extension)

| Run | Generator / judge | Judge input | Approval |
|---|---|---|---|
| `results/full_dev.jsonl` (v1 prompt) | llama3.2:3b / qwen2.5:3b | SQL + explanation (no schema) | 15/50 (30%) |
| `results/full_test.jsonl` (v1 prompt) | llama3.2:3b / qwen2.5:3b | no schema | 36/116 (31%) |
| `results/v2_dev.jsonl` (v2 prompt) | not recorded | no schema | 30/50 (60%) |
| `results/improved_dev.jsonl` | not recorded | no schema | 34/50 (68%) |

These are LLM-judge approval rates with the original judge and were not re-scored. In our re-run with the same original judge prompt (no schema), the v2 baseline received 24/50 (48%); the difference to the recorded 60% may come from a different generator, model versions or Ollama defaults, which the historical files do not record.

## 10. Threats to validity

* **No human labels.** The main quality comparison rests on AI-assisted labels; one annotator, partly not blind.
* **Small samples.** 50 dev pairs; only the v3 difference approaches significance (p = 0.053).
* **Design on dev.** v4 was designed after inspecting dev errors; the held-out test sample (section 7) is the fair estimate for v4.
* **Small generator.** A 3B model may misuse retrieved context in ways larger models do not; the result may not transfer to `qwen2.5:7b` or hosted models.
* **Judge weakness.** The 3B rubric judge is at chance against the reference labels; LLM-judge approvals alone would have hidden the v3 regression entirely.
* **Dataset artefacts.** The 166 queries were generated by an LLM and include odd joins (for example `clients.id = transactions.account_id`), which reward literal description over plausible interpretation.
* **Relevance map** for retrieval metrics is AI-written and section-level.

## 11. Reproducing

```bash
python -m sqlrag ingest
python -m sqlrag retrieval-eval --split all --chunk-sizes 400,200,300,500,600
SQLRAG_NUM_GPU=99 SQLRAG_RAG_PROMPT=prompts/v3_rag.md SQLRAG_TOP_K=6 SQLRAG_MAX_CONTEXT_TOKENS=1800 SQLRAG_FILL_CONTEXT=1 \
  python -m sqlrag experiment dev rag_vs_baseline_dev --generator-model llama3.2:3b --judge-model qwen2.5:3b
python -m sqlrag experiment dev rag_vs_baseline_dev --phase judge --judge-model qwen2.5:7b
python -m sqlrag experiment dev rag_vs_baseline_dev --phase judge --judge-model qwen2.5:3b --judge-mode legacy
python -m sqlrag experiment abstention_probe abstention_probe --generator-model llama3.2:3b
python -m sqlrag summarize results/rag/rag_vs_baseline_dev
python -m sqlrag agreement
```

The v4 experiment reused the baseline rows of `rag_vs_baseline_dev` and ran `--conditions rag` with the v4 defaults.
