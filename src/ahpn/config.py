API_BASE_URL = "https://ahpn-api-prod.gsc1-pub.lib.utexas.edu"
SITE_BASE_URL = "https://ahpn.lib.utexas.edu"

# One-page public document used as the first scrape target.
EXAMPLE_DOCUMENT_ID = 2127283

USER_AGENT = "ahpn-scraper/0.1 (AHPN public-archive research)"
DEFAULT_REQUEST_DELAY_SECONDS = 1.0

DEFAULT_DATABASE_URL = "postgresql+psycopg://ahpn:ahpn@localhost:5433/ahpn"
DEFAULT_CHUNK_CHARS = 1500
DEFAULT_CHUNK_OVERLAP = 200

# Synchronous DetectDocumentText accepts a single image up to 10 MB.
TEXTRACT_MAX_BYTES = 10 * 1024 * 1024

# First-pass retrieval: BM25 shortlist, then Cohere Rerank on Bedrock.
DEFAULT_RERANK_MODEL = "cohere.rerank-v3-5:0"
DEFAULT_SEARCH_CANDIDATES = 100
DEFAULT_RERANK_CANDIDATES = 50
DEFAULT_SEARCH_K = 10
