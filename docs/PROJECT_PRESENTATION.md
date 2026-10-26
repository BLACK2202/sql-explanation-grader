# Project presentation outline

A 12–15 minute talk with a live demo. Each slide lists what to show and what to say. Numbers come from `docs/EVALUATION.md`.

---

### Slide 1. Title

**SQL Explanation Grader: RAG Enhanced.** Grounded SQL explanations with citations, abstention and an honest baseline-vs-RAG evaluation.

### Slide 2. The problem

* People who do not know SQL need to understand what a query returns.
* Small local LLMs explain fluently but get details wrong (left joins, RANK ties, NULL) and invent meaning ("doctors who treated these patients").
* The original project graded explanations with an LLM judge that did not see the table definitions.

### Slide 3. What existed

166 SQLite queries in six domains; a generator (Ollama) and a judge; prompts v1 and v2; dashboard and PDF reports. Recorded judge approval: 30% (v1), 60% (v2), 68% (improved).

### Slide 4. Research question

Does retrieving official SQL documentation before generating (RAG) make explanations more correct, complete and reliable than the prompt-only approach?

### Slide 5. Architecture

Show the diagram from the README. Three parts: ingestion (parse, chunk, embed, index), explanation (analyse SQL, retrieve, check evidence, generate with citations), evaluation (judges, rules, reference labels, statistics).

### Slide 6. Corpus and ingestion

* 15 documentation pages: SQLite (public domain) and PostgreSQL tutorials (PostgreSQL License); manifest with URL, license, date, hashes.
* Parse HTML/PDF/Markdown/text, keep headings, never split SQL examples, 300-token chunks with 60-token overlap.
* bge-small embeddings in ChromaDB plus a BM25 index; ingestion is idempotent and detects duplicates.

### Slide 7. Retrieval design (measured, not assumed)

* SQLGlot turns the query into a list of features; one search per feature.
* Dense + BM25 fused with Reciprocal Rank Fusion.
* Ablation over 166 queries: one query per feature 0.89 context recall vs 0.80 for the raw SQL; chunks above 400 tokens overflow the embedder's 512-token window; BM25 alone was slightly better than hybrid on context recall (we say so).

### Slide 8. Grounding and abstention

* Before writing, every feature must be covered by a retrieved passage; otherwise the system abstains (or answers with an explicit "not grounded" label).
* Probe: 10 queries with undocumented JSON/math functions, 10 documented: 10/10 abstained, 10/10 answered, no LLM call needed for the refusals.
* Adding the JSON documentation page (9 s) made the JSON queries answerable.

### Slide 9. Live demo (dashboard)

1. Playground → load the LEFT JOIN example → RAG → show citations, coverage ticks, passages and scores.
2. Load an abstention probe (JSON) → show the refusal and the missing feature.
3. Compare both on a RANK query.
4. Knowledge Base → show documents and licenses; search "how does RANK handle ties".

### Slide 10. Experiment setup

Same 50 dev queries, same generator (llama3.2:3b), temperature 0, seed 42, same judges, alternating order; the RAG prompt is the baseline prompt plus a section on using sources. Two RAG designs: v3 (6 passages) and v4 (one passage per feature).

### Slide 11. Results: quality

| | Baseline | RAG v3 | RAG v4 |
|---|---|---|---|
| Reference labels "good" (AI-assisted) | 20/50 | 10/50 | 16/50 |
| Paired difference vs baseline | | −20 pts (p = 0.053) | −8 pts (p = 0.54) |
| Test, held out (random 40, blind) | 16/40 | | 11/40 (−12.5 pts, p = 0.27) |

Say plainly: with a 3B generator, RAG did not improve correctness; the first design made it worse, and the held-out test confirms the focused design is not better.

### Slide 12. Why RAG v3 lost

Examples from the label notes: an inner join described as a "cartesian product", "compound SELECT" for a single query, "books has 3 columns", raw SQL keywords in 33/50 answers (baseline 5/50). The model pasted general documentation instead of explaining this query. The focused v4 design reduced the context from about 1,450 to 490 tokens and recovered most of the loss.

### Slide 13. Can we trust the judge?

| Judge | Agreement with reference labels (κ) |
|---|---|
| qwen2.5:3b, new rubric + schema | 0.01 (approves 99%) |
| qwen2.5:3b, original prompt | 0.20 |
| qwen2.5:7b, rubric + schema | 0.26 (approves 71%, reference 30%) |

Lesson: LLM-judge approval alone would have hidden the v3 regression. Measure the judge before trusting it.

### Slide 14. Engineering

FastAPI with OpenAPI docs, dashboard, 130 automated tests (no models needed), GitHub Actions, Docker compose with Ollama, safe SQL handling (compile-only, authorizer), untrusted-document handling.

### Slide 15. Limitations and next steps

No human labels yet (labelling page ready); small samples; small generator; lexical evidence checks; small corpus. Next: human labels, a larger generator, retrieve-then-verify generation.

### Slide 16. Conclusion

RAG gave the system verifiable sources, evidence checks and safe refusals. It did not, on its own, make a 3B model's explanations more correct; how the retrieved text is used matters more than retrieving it. The evaluation infrastructure is the main reusable result.

---

## Likely questions

See [TEACHER_QA.md](TEACHER_QA.md). Prepare especially: why the judge disagrees with the labels, why hybrid is not always best, why v4 is not a fair final result on dev, and what you would change with a bigger GPU.
