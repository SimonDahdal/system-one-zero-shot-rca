"""Edit experiment, model and telemetry settings here."""
from pathlib import Path

DATA_DIR = Path("data")
DATASETS = ["RE2-OB", "RE2-SS", "RE2-TT"]
RANDOM_STATE = 42

KEV_URL = "http://127.0.0.1:8009"
# KEV_URL = "http://127.0.0.1:8009"
KEV_MODEL = "kev-latest"

LAYA_URL = "http://127.0.0.1:8010/v1/systemone"
LAYA_TIMEOUT_SECONDS = 120
PRINT_LAYA_INPUT = False

NIMBLE_BASE_URL = "http://127.0.0.1:8011"
NIMBLE_TIMEOUT_SECONDS = 300
PRINT_NIMBLE_INPUT = False
# Retry only explicit context-length rejections; never change the choice schema.
NIMBLE_CONTEXT_RETRIES = 5

SEMIF_BASE_URL = "http://127.0.0.1:8012"
SEMIF_TIMEOUT_SECONDS = 300
PRINT_SEMIF_INPUT = False
SEMIF_CONTEXT_RETRIES = 5

RIZZO_BASE_URL = "http://127.0.0.1:8013"
RIZZO_MODEL = "rizzo-latest"
RIZZO_TIMEOUT_SECONDS = 300
PRINT_RIZZO_INPUT = False
RIZZO_CONTEXT_RETRIES = 5

VON_BASE_URL = "http://127.0.0.1:8014"
VON_MODEL = "von-latest"
VON_TIMEOUT_SECONDS = 300
PRINT_VON_INPUT = False
VON_CONTEXT_RETRIES = 5

NANOJEV_BASE_URL = "http://127.0.0.1:8015"
NANOJEV_TIMEOUT_SECONDS = 300
PRINT_NANOJEV_INPUT = False
NANOJEV_CONTEXT_RETRIES = 5

D1_BASE_URL = "http://127.0.0.1:8016"
D1_MODEL = "LiquidAI/d1-3B"
D1_TIMEOUT_SECONDS = 300
PRINT_D1_INPUT = False
D1_CONTEXT_RETRIES = 5

DECIDER_BASE_URL = "http://127.0.0.1:8018"
DECIDER_MODEL = "Mapika/decider-2b"
DECIDER_TIMEOUT_SECONDS = 300
PRINT_DECIDER_INPUT = False
DECIDER_CONTEXT_RETRIES = 5

H2O_LIGHTNING_BASE_URL = "http://127.0.0.1:8019"
H2O_LIGHTNING_MODEL = "h2oai/h2o-lightning-4b"
H2O_LIGHTNING_TIMEOUT_SECONDS = 300
PRINT_H2O_LIGHTNING_INPUT = False
H2O_LIGHTNING_CONTEXT_RETRIES = 5

NEOHORSE_BASE_URL = "http://127.0.0.1:8020"
NEOHORSE_MODEL = "TokenRhythm/NeoHorse-Jev-4B"
NEOHORSE_TIMEOUT_SECONDS = 300
PRINT_NEOHORSE_INPUT = False
NEOHORSE_CONTEXT_RETRIES = 5

D3_FLASH_BASE_URL = "http://127.0.0.1:8021"
D3_FLASH_MODEL = "vllm-sr/d3-flash"
D3_FLASH_TIMEOUT_SECONDS = 300
PRINT_D3_FLASH_INPUT = False
D3_FLASH_CONTEXT_RETRIES = 5

# Independent telemetry windows and prompt budgets. Set a window to None to
# use all available data on that side of the injection boundary.
METRIC_WINDOW_SECONDS = 150
BEFORE_LOG_WINDOW_SECONDS = 50
AFTER_LOG_WINDOW_SECONDS = 80
BEFORE_LOG_LINES = 30
AFTER_LOG_LINES = 70
MAX_LOGS_PER_CONTAINER = 3
MAX_LOG_MESSAGE_CHARS = 700
MAX_METRIC_ANOMALIES = 5
MAX_METRICS_PER_SERVICE = 5
MAX_TRACE_EDGES = 10
PRINT_KEV_INPUT = False
INCLUDE_METRIC_SUMMARY = True
INCLUDE_TRACE_SUMMARY = True
MAX_LOG_STAT_CONTAINERS = 20

# RE2-TT contains many more services and trace edges than RE2-OB/RE2-SS.
# Give it a smaller prompt budget and skip its expensive trace aggregation.
SYSTEM_TELEMETRY_CONFIG = {
    "RE2-OB": {},
    "RE2-SS": {},
    "RE2-TT": {
        "before_window_seconds": 15,
        "after_window_seconds": 45,
        "before_log_lines": 2,
        "after_log_lines": 6,
        "max_logs_per_container": 2,
        "max_log_message_chars": 180,
        "max_log_stat_containers": 8,
        "max_metric_anomalies": 2,
        "include_trace_summary": False,
    },
}

# Preserve a reproducible random split.
TEST_SIZE = 0.50

# Prepared data includes the feature matrix, incident metadata and shared split.
PREPARED_DATA_PATH = Path("artifacts/re2_dataset.joblib")
RESULTS_DIR = Path("results")
DEFAULT_MODELS = ("rf", "kev")
# Wait for manual VRAM unloading/deployment before starting the next model.
PAUSE_BETWEEN_MODELS = True
RF_N_ESTIMATORS = 300
RF_MAX_FEATURES = "sqrt"
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1

def telemetry_config(system):
    """Return global defaults with optional per-system overrides."""
    config = {
        "before_window_seconds": BEFORE_LOG_WINDOW_SECONDS,
        "after_window_seconds": AFTER_LOG_WINDOW_SECONDS,
        "before_log_lines": BEFORE_LOG_LINES,
        "after_log_lines": AFTER_LOG_LINES,
        "max_logs_per_container": MAX_LOGS_PER_CONTAINER,
        "max_log_message_chars": MAX_LOG_MESSAGE_CHARS,
        "max_log_stat_containers": MAX_LOG_STAT_CONTAINERS,
        "max_metric_anomalies": MAX_METRIC_ANOMALIES,
        "include_trace_summary": INCLUDE_TRACE_SUMMARY,
    }
    config.update(SYSTEM_TELEMETRY_CONFIG.get(system, {}))
    return config


SYSTEM_CONTEXT = {
    "RE2-OB": """
System: Online Boutique, an e-commerce microservice application.
Relevant service roles:
- checkoutservice orchestrates checkout. It reads the cart and product data,
  converts prices, and coordinates payment, shipping, and confirmation email.
- productcatalogservice owns the product catalogue and product lookup/search.
- currencyservice supplies exchange rates and performs currency conversion.
- recommendationservice recommends products, using catalogue information.
- emailservice sends order confirmation emails near the end of checkout.
Important causal relationships:
- checkoutservice is a high-level caller of several other services, so its
  errors may be symptoms of failures in a downstream dependency.
- recommendationservice can surface productcatalogservice failures.
- A currencyservice failure commonly propagates into price display or checkout.
""",
    "RE2-SS": """
System: Sock Shop, an e-commerce microservice application.
Relevant service roles:
- catalogue owns sock/product descriptions, inventory-facing catalogue data,
  search, and product lookup.
- carts owns each user's shopping cart and cart items.
- user owns customer accounts, login information, and addresses/cards.
- orders creates and retrieves orders and coordinates the order workflow.
- payment authorizes payment requests made during order placement.
Important causal relationships:
- orders is an orchestrator and depends on data or responses from carts, user,
  payment, and other fulfilment services. Its errors can therefore be secondary.
- Front-end or API errors may simply report an unavailable domain service.
- Failures in carts, user, catalogue, or payment can propagate into orders.
""",
    "RE2-TT": """
System: Train Ticket, a railway booking microservice application.
Relevant service roles:
- ts-auth-service authenticates users and validates access credentials/tokens.
- ts-order-service creates, stores, retrieves, and changes ticket orders.
- ts-route-service owns route definitions and route-related lookup data.
- ts-train-service owns train identifiers and train configuration information.
- ts-travel-service handles travel queries and composes trip options using
  route, train, station, price, and related service data.
Important causal relationships:
- ts-travel-service is an aggregation/orchestration layer; errors there may be
  caused by ts-route-service, ts-train-service, or another queried dependency.
- ts-order-service participates later in the booking flow and can report
  upstream authentication or travel-data problems as well as its own failures.
- Authentication failures can affect many callers without those callers being
  the original faulty service.
""",
}


