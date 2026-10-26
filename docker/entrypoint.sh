#!/bin/sh
# Index the corpus (idempotent: unchanged documents are skipped), then serve.
set -e
python -m sqlrag ingest > /tmp/ingest.log 2>&1 || { cat /tmp/ingest.log; echo "ingestion failed"; exit 1; }
tail -n 25 /tmp/ingest.log
exec python -m sqlrag serve --host 0.0.0.0 --port 8000
