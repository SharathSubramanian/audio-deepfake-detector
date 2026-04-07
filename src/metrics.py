from prometheus_client import Counter, Histogram, REGISTRY

# =========================
# SAFE METRIC FETCH / CREATE
# =========================
def get_metric(name, desc, metric_type="counter", labels=None):
    """
    Safely get or create a Prometheus metric.
    Prevents duplicate registration errors in Streamlit.
    """

    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]

    if metric_type == "counter":
        return Counter(name, desc, labels or [])
    elif metric_type == "histogram":
        return Histogram(name, desc, labels or [])
    else:
        raise ValueError("Unsupported metric type")


# =========================
# METRICS
# =========================

PREDICTIONS = get_metric(
    "model_predictions_total",
    "Total predictions made",
    metric_type="counter",
    labels=["model", "prediction"]
)

ERRORS = get_metric(
    "model_prediction_errors_total",
    "Total prediction errors",
    metric_type="counter"
)

LATENCY = get_metric(
    "model_prediction_latency_seconds",
    "Time taken for prediction",
    metric_type="histogram"
)

CONFIDENCE = get_metric(
    "model_prediction_confidence",
    "Prediction confidence distribution",
    metric_type="histogram"
)