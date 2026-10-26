# Reference labels

Reference labels say whether an explanation is really **good** or **bad**. They are used to measure how far the LLM judge can be trusted (`python -m sqlrag agreement`, Evaluation tab of the dashboard).

## Two kinds of labels, never mixed

| File | `label_source` | Who labelled | Status |
|---|---|---|---|
| `human_labels.jsonl` | `human` | People, through the dashboard's Human labelling tab (blind to the condition) | None yet. Created on first use. |
| `ai_reference_labels.jsonl` | `ai_reference` | Claude (an AI assistant) | 280 labels: `rag_vs_baseline_dev` all 100 (blind); `rag_v4_dev` 50 RAG answers (not blind) + the 50 shared baseline answers (copied); `rag_v4_test` random 40 of 116 queries, 80 answers (blind, seed 2026) |

AI-assisted labels are a careful second opinion from a stronger model, **not human ground truth**. Every report shows the two sources separately. The best next step for the project is to have one or two people label (or confirm) the same explanations in the dashboard, then compare human labels with both the LLM judge and the AI-assisted labels.

Each label stores the SHA-256 prefix of the exact explanation text it refers to. A label whose hash does not match the stored explanation is ignored, so labels can never be silently applied to different text.

## Decision rules

Based on `data/LABELLING_GUIDE.md` (good only if correct, complete and clear), applied as follows:

An explanation is **bad** if any of these holds:

1. **Wrong claim** about this query and schema: wrong rows or values returned, wrong sort direction, a "most recent" or "top" claim the ordering does not support, wrong join behaviour (for example calling an inner join a cartesian product, or saying unmatched rows are kept), wrong tie behaviour of RANK or ROW_NUMBER, wrong number of columns, a filter that does not exist, or a statement that the query returns a single value when it returns rows.
2. **Invented meaning**: a relationship or business interpretation the schema does not state (for example "doctors who treated the patients", "students" counted when no student data is involved, "joined the library first" for `id = 1`).
3. **Missing operation**: a returned column, aggregate result, join, filter, grouping, ordering or LIMIT is not described.
4. **Not clear**: most of the text is documentation jargon or internal processing (input data, result expression list, cartesian product) that a non-technical reader cannot follow.

Tolerated (still **good**): a single irrelevant but harmless sentence; a short technical term that is explained; a loose but not wrong description of SQLite's "bare column" behaviour.

## Procedure used for the AI-assisted labels

1. All explanations of an experiment were written to a sheet in random order, with citation markers removed and the condition (baseline or RAG) replaced by an opaque key.
2. Each explanation was read against its schema and SQL and labelled with the rules above, with a one-line reason (`notes`).
3. The key was used afterwards to attach the item id and condition.

Because RAG answers often use documentation wording, a reader can sometimes guess the condition; blinding reduces but does not remove this bias.
