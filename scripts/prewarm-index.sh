#!/usr/bin/env bash
# Build the filing index that ships inside the container image.
#
# The deployed demo runs with FINAGENT_RAG_INDEX_ON_DEMAND=0, so filing_search
# can only answer for filings indexed here. Run this before `fly deploy` — the
# Dockerfile copies the resulting .chroma/ directory into the image.
#
#   ./scripts/prewarm-index.sh
#   TICKERS="AAPL NVDA" FORM_TYPES="10-K 10-Q" ./scripts/prewarm-index.sh
set -euo pipefail

TICKERS=${TICKERS:-"AAPL NVDA MSFT TSLA AMZN GOOGL META"}
FORM_TYPES=${FORM_TYPES:-"10-K"}

cd "$(dirname "$0")/.."

for ticker in $TICKERS; do
  for form in $FORM_TYPES; do
    echo "==> indexing $ticker $form"
    # A failure here is usually one filing SEC served oddly; keep going so a
    # single bad document doesn't cost the whole index.
    uv run finagent index-filings "$ticker" --form-type "$form" || echo "!! skipped $ticker $form"
    # SEC asks automated clients to stay under 10 requests/second.
    sleep 1
  done
done

echo
echo "index ready: $(du -sh .chroma | cut -f1) in .chroma/"
