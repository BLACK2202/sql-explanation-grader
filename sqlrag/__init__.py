"""sqlrag: Retrieval-Augmented Generation extension of the SQL Explanation Grader.

The package is layered so each stage can be tested on its own:

    ingest/       parse -> clean -> chunk documents (no ML dependencies)
    index/        embeddings, ChromaDB vector store, BM25 keyword index
    retrieval     dense + BM25 + Reciprocal Rank Fusion, context assembly
    sql_analysis  SQLGlot-based structural analysis and retrieval queries
    grounding     evidence sufficiency, citation validation, abstention
    explain       baseline and RAG explanation generation
    judge         schema-aware rubric judge (keeps the legacy good/bad grade)
    checks        deterministic, model-free checks on explanations
    evaluation/   retrieval metrics, judge agreement, baseline-vs-RAG experiments
    api           FastAPI application serving the dashboard and JSON endpoints
"""

__version__ = "0.3.0"
