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

# Retrieval defaults. Cohere Embed v4 / Titan v2 both support 1024 dims on Bedrock.
DEFAULT_EMBEDDING_MODEL = "cohere.embed-v4:0"
DEFAULT_EMBEDDING_DIMS = 1024
FALLBACK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_JUDGE_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"
DEFAULT_QUERY_GEN_MODEL = "anthropic.claude-sonnet-4-20250514-v1:0"

RRF_K = 60
DEFAULT_SEARCH_CANDIDATES = 100
DEFAULT_SEARCH_K = 10

# Approximate HNSW defaults; tune with `ahpn eval scale`.
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 64
DEFAULT_HNSW_EF_SEARCH = 40
