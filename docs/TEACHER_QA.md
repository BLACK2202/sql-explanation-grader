# Teacher Q&A: the project in simple words

This guide prepares you to answer questions about the project in a presentation or a defence. Each answer is short and uses simple English. Where a number is quoted, it comes from our own measurements in `docs/EVALUATION.md`.

---

## Part 1: The project

**What does the project do?**
A user gives an SQL query and the database's table definitions. The system writes a short, plain-English explanation of what the query returns, for someone who does not know SQL. A second AI model then grades that explanation.

**What did the original project already have?**
166 SQL queries in six domains (shop, school, hospital, bank, library, hotel), a local AI that writes explanations, a second local AI that grades them good or bad (the "judge"), two prompt versions, a live dashboard and PDF reports.

**What did we add?**
Retrieval-Augmented Generation (RAG): before writing an explanation, the system looks up the relevant parts of real SQL documentation and gives them to the AI. We also improved the judge, added honest evaluation, an API, a new dashboard, tests and Docker.

**What is the research question?**
Does looking up documentation first (RAG) make the explanations more correct, more complete and more reliable than the original prompt-only approach?

---

## Part 2: The basic ideas

**What is an LLM?**
A Large Language Model is a program trained on a huge amount of text. It predicts the next word again and again, which lets it write answers. Examples: Llama, Qwen, GPT, Claude. We use small ones (3 billion parameters) that run on a laptop.

**What is a prompt?**
The instructions and the input we send to the LLM. Our prompts are files in `prompts/`, for example `v2.md` (baseline) and `v3_rag.md` (RAG).

**What is hallucination?**
When an AI states something that sounds right but is not supported by the facts. Example from our data: for a query that only compares doctor names with patient names, the AI wrote that the doctors "have treated" those patients. Nothing in the query or the tables says that.

**What is RAG?**
Retrieval-Augmented Generation means "look it up, then answer". Step 1 (retrieval): find the most relevant passages in a collection of trusted documents. Step 2 (generation): give those passages to the LLM together with the question and ask it to answer using them and to cite them.

**Why do we need retrieval? Doesn't the LLM already know SQL?**
It knows a lot, but small models make mistakes on details: what a LEFT JOIN does with rows that have no match, whether BETWEEN includes both ends, how RANK handles ties, what happens with NULL. Retrieval gives the model the exact rule from the official documentation at the moment it writes. It also lets us show the user where each claim came from, and lets us refuse to answer when we have no documentation for something.

**Is RAG the same as training the model?**
No. Training changes the model's internal numbers and is expensive. RAG does not change the model at all; it only changes what we put in the prompt. Adding a new document takes seconds.

---

## Part 3: The pipeline, piece by piece

**What is the corpus?**
The collection of documents we search. Ours has 15 pages: 10 official SQLite pages (public domain) and 5 PostgreSQL tutorial pages (PostgreSQL License). Every document is listed in `corpus/manifest.json` with its URL, license and date. One more page (JSON functions) is kept aside on purpose to demonstrate abstention.

**Why did you record licenses?**
Because we copy text from websites into our system and into prompts. We only used sources whose license allows reuse, we respected `robots.txt`, and we kept the required copyright notice (`corpus/LICENSES.md`).

**What is parsing?**
Turning a file (HTML web page, PDF, Markdown, text) into clean text while keeping its structure: headings, paragraphs, code examples and page numbers. We remove menus, footers and scripts.

**What is chunking?**
Cutting long documents into small pieces ("chunks"). A whole page is too long to give to the model and too broad to match a specific question. We use chunks of about 300 tokens (around 200 words), and neighbouring chunks overlap by about 60 tokens so a sentence cut at the border is not lost. We never cut an SQL code example in half.

**What is a token?**
A small piece of text, roughly a short word or part of a word. Models measure input and output length in tokens.

**Why 300 tokens per chunk?**
Our embedding model can only read 512 tokens at a time; longer text is cut off. We tested 200, 300, 400, 500 and 600. At 500 and 600, about 40% of chunks were too long and retrieval got worse. 300 kept every chunk readable and worked well.

**What are embeddings?**
An embedding is a list of numbers (for us, 384 numbers) that represents the meaning of a piece of text. Texts with similar meaning get similar lists of numbers. "Rows with no matching partner" and "unmatched rows in an outer join" end up close together even though they share few words.

**Does the LLM read the embeddings?**
No. Embeddings are only used to find passages. The LLM receives the passages as normal text.

**What is the difference between an LLM and an embedding model?**
An embedding model turns text into numbers so we can compare meanings; it cannot write. An LLM writes text. Ours are different programs: the embedding model is BGE-small (33 million parameters, runs on the CPU); the LLMs are Llama 3.2 3B and Qwen 2.5 (3 billion and 7 billion parameters, run with Ollama).

**What is a vector database? Why ChromaDB?**
A database that stores the number lists (vectors) with their text and can quickly find the vectors closest to a query vector. ChromaDB is free, runs inside our Python program, and saves to disk (`rag_store/`), so we do not rebuild the index each time.

**What is BM25?**
A classic keyword search formula (used by search engines for decades). A passage scores high when it contains the query's words, especially rare words. It is good at exact names like `json_extract` or `HAVING`, where embeddings can be vague.

**What is hybrid retrieval?**
Using both searches (meaning-based and keyword-based) and combining their rankings. We combine them with Reciprocal Rank Fusion: a passage gets 1/(60 + its rank) from each list, and the scores are added. A passage that is high in both lists wins. In our test, keyword search alone was slightly better on one metric and meaning search on another, so we report all three honestly.

**How do you turn an SQL query into a search?**
An SQL query is not a good search phrase. We use SQLGlot, a library that understands SQL grammar, to list the features the query uses (LEFT JOIN, IS NULL, GROUP BY, window functions, ...). Each feature becomes its own search sentence, for example "LEFT JOIN keeps every row of the left table; unmatched columns are NULL". This worked much better than searching with the raw SQL.

**What is SQLGlot?**
A Python library that reads SQL and turns it into a tree of its parts. We use it to detect features, read the LIMIT number and the sort direction, and list the tables.

**What is Ollama?**
A free program that downloads and runs LLMs on your own computer and offers them through a small web API (`http://localhost:11434`). No paid API key and no data leaves the machine.

---

## Part 4: Trust and safety

**How do you stop the model from inventing things?**
Several layers. The prompt tells it to describe only what the SQL says and never invent values. It must cite the passage it used, like `[S2]`, and we check every citation exists. Before writing, we check that the retrieved passages cover every SQL feature in the query. If they do not, the system refuses instead of guessing.

**What is abstention?**
Refusing to answer. If the query uses a feature we have no documentation for (for example JSON or math functions in our default corpus), the system says "not enough supporting documentation" and names the missing feature. It never pretends a guess is grounded. There is also a "fallback" setting that answers without sources but labels the answer clearly as not grounded.

**How do you detect unsupported answers?**
Four ways: (1) the judge lists claims not supported by the SQL and schema; (2) citation checks: a citation must point to a passage that was actually given, and the cited passage should share the sentence's key words; (3) rule-based checks, for example a number in the explanation that appears nowhere in the SQL is flagged as invented; (4) the evidence-coverage check before writing.

**Can a document trick the model ("prompt injection")?**
We treat documents as untrusted. They are wrapped in `<source>` tags, any tag-like text inside them is neutralised, and the prompt says sources are reference material, not instructions. Uploaded files are only read as text, never run. SQL from users is only checked (compiled), never run, and dangerous commands are blocked.

**Which is the authority, the SQL or the documentation?**
The SQL. Documentation explains how a feature behaves in general; the query decides what this particular query does.

---

## Part 5: Evaluation

**Why do we need two LLM roles?**
One model writes (the generator), another grades (the judge). Grading 166 explanations by hand many times is slow, so an automatic judge lets us compare versions quickly. Using a different model as judge reduces the risk that a model approves its own style.

**What does the judge evaluate?**
It reads the table definitions, the SQL, a list of the operations the query contains, and the explanation. It scores correctness, completeness, clarity and groundedness from 1 to 5, lists unsupported claims, and gives the old good/bad grade so we can compare with earlier results.

**What was wrong with the old judge?**
It did not see the table definitions, so it could not check whether column or table descriptions were right. We fixed that: the judge now receives the schema. Be honest about the result, though: when we tested the small 3B judge with and without the schema against the reference labels, adding the schema did not make its grades more accurate (agreement went from slight to chance level). The fix is necessary for a judge to be able to check the schema, but a small model does not use the extra information well. The 7B judge agreed better than any 3B setting.

**Is the judge always right?**
No. It is also an AI. That is why we call its result "judge approval", not "accuracy". We measure how often it agrees with reference labels and report that next to the results. In our test the small 3B judge with the new rubric approved 99 of 100 explanations, although the reference labels rated only 30 as good: its agreement was at chance level (Cohen's kappa 0.01). That is a finding in itself.

**What are reference labels? Are they human?**
Labels that say whether an explanation is really good or bad. The dashboard has a "Human labelling" page where a person labels explanations without being told which system wrote them. Labels made by people are stored as `human`. The project also contains a set of labels drafted by an AI assistant (Claude) after careful reading; these are stored as `ai_reference` and are always reported separately, never called human.

**What are accuracy, precision, recall and F1?**
When we compare the judge's good/bad decisions with reference labels:
* accuracy = share of decisions that match;
* precision (good) = of the explanations the judge called good, how many really are good;
* recall (good) = of the really good explanations, how many the judge called good;
* recall (bad) = of the really bad explanations, how many the judge caught;
* F1 = a balance of precision and recall;
* Cohen's kappa = agreement corrected for luck (0 = chance, 1 = perfect).
A confusion matrix is the 2×2 table of these counts.

**What are Recall@k and MRR?**
Retrieval metrics. Recall@5: of the SQL features in a query, how many have a relevant passage among the top 5 results. MRR: 1 divided by the position of the first relevant passage (1.0 if it is first, 0.5 if second). We also measure "context recall": the same idea for the passages actually given to the model.

**Why compare baseline and RAG?**
To know whether RAG helps, we must compare it with the same system without RAG. Both use the same 50 queries, the same generator model, the same settings (temperature 0, same seed, same context size), the same judge and the same machine. The only difference is the retrieved documentation and the instructions to use it.

**Why is the same item compared in pairs?**
Because some queries are harder than others. Comparing each query's baseline and RAG answers directly removes that difference. We use the McNemar test (it looks only at queries where the two systems disagree) and a bootstrap confidence interval.

**What is a confidence interval?**
A range that shows how sure we are. "+6 points (95% interval −4 to +16)" means the true difference could plausibly be anywhere from −4 to +16; because the range includes 0, we cannot claim an improvement.

**Did RAG improve the explanations?**
Not with our small 3B generator. On the 50 development queries, the reference labels rated 20 baseline explanations good, 10 from the first RAG design (v3) and 16 from the focused design (v4). The first design was worse because the model copied general documentation sentences that were wrong for the query (for example calling an inner join a "cartesian product"). The focused design gave the model fewer, more relevant passages and recovered most of the loss, but did not beat the baseline. With 50 queries these differences are uncertain (p = 0.053 and p = 0.54). Because we designed v4 after looking at the development errors, we also tested it once on 40 randomly chosen queries from the untouched test split: 16 of 40 baseline and 11 of 40 RAG explanations were good, so the conclusion held. What RAG did deliver: every citation pointed to a real passage, and the system correctly refused all 10 questions about features it had no documentation for.

**Why can RAG make answers worse?**
More text in the prompt is not automatically better. A small model tends to repeat what it reads. When the passages describe general rules (how all joins work internally, compound queries, OFFSET), the model mixed those general rules into the explanation of a specific query where they do not apply. Giving it only one focused passage per feature, and telling it to use sources only to check behaviour, reduced this.

---

## Part 6: Engineering

**What is FastAPI? What is an API endpoint?**
FastAPI is a Python library for building web APIs. An endpoint is a web address that does one job, for example `POST /api/explain` takes SQL and returns an explanation. FastAPI also creates interactive documentation at `/docs`.

**What is Pydantic?**
A library that checks data has the right shape. We use it to validate requests and to force the LLM's answer into valid JSON with the expected fields.

**What is Docker?**
A tool that packages the program with everything it needs so it runs the same on any computer. `docker compose up` starts two containers: Ollama (the LLMs) and our app.

**What tests did you write?**
130 automatic tests that run without any AI model (they use a fake model and simple fake embeddings): parsing, chunking, indexing, duplicates, search, SQL analysis, safety checks, citations, abstention, the judge, statistics, the API, and checks that the original project and its recorded results are unchanged. Three extra tests use the real Ollama models and run only on demand. GitHub Actions runs the tests on every push.

**What does "idempotent" mean?**
Doing it twice gives the same result as doing it once. Running ingestion again does not create duplicate chunks.

---

## Part 7: Limitations (be ready to say these yourself)

* The judge is an AI and was lenient; its approvals are not ground truth.
* There are no human labels yet; the reference labels are AI-assisted and labelled as such.
* 50 development queries are few; small differences are within the noise.
* The relevance map used to score retrieval was written by an AI assistant from the section headings and has not been reviewed by a person.
* Small 3B models were used because of the laptop's 4 GB GPU; results may differ with larger models.
* RAG answers were slower (more text to read) and sometimes more technical, because the model copied documentation words.
* The rule-based checks use word lists and can miss paraphrases.
* The corpus is small (15 pages) and SQLite-focused; PostgreSQL tutorial pages differ slightly in dialect.
