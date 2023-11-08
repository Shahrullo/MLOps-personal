# Import libraries
import logging
from typing import Any, Dict

import pandas as pd
import prometheus_client
from evidently.model_monitoring import (
    DataDriftMonitor,
    DataQualityMonitor,
    ModelMonitoring,
    NumTargetDriftMonitor,
    RegressionPerformanceMonitor,
)
from evidently.pipeline.column_mapping import ColumnMapping
from flask import Flask, Response

app = Flask(__name__)

# Define the column mapping
column_mapping = ColumnMapping()
column_mapping.target = (
    "target"  # 'target' is the name of the column with the target function
)
column_mapping.prediction = (
    "prediction"  # 'prediction' is the name of the column(s) with model predictions
)
column_mapping.id = None  # There is no ID column in the dataset

column_mapping.numerical_features = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']  # List of numerical features
column_mapping.categorical_features = None # There is no categorical features


@app.route("/metrics")
def display_metrics():

    """
    Displays data metrics on the web page.
    """

    res = []
    for _, value in metrics.items():
        res.append(prometheus_client.generate_latest(value))
    return Response(res, mimetype="text/plain")


def monitor_evidently() -> Dict[str, Any]:

    """
    Initiates Evidently monitoring, and computes the data metrics.
    """

    local_path = "../../"

    reference = pd.read_csv(f"{local_path}data/reference.csv")
    current = pd.read_csv(f"{local_path}data/current.csv")

    data_metrics = {}
    registry = prometheus_client.CollectorRegistry()

    # Monitoring program
    evidently_monitoring = ModelMonitoring(
        monitors=[
            NumTargetDriftMonitor(),
            DataDriftMonitor(),
            RegressionPerformanceMonitor(),
            DataQualityMonitor(),
        ],
        options=None,
    )

    # Monitoring results
    evidently_monitoring.execute(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    results = evidently_monitoring.metrics()
    logging.info("Data metrics were found by Evidently.")

    for i, (metric, value, labels) in enumerate(results, start=1):

        if labels:
            label = "_".join(list(labels.values()))
        else:
            label = "na"

        metric_key = f"evidently:{metric.name}:{label}"
        prom_metric = prometheus_client.Gauge(metric_key, "", registry=registry)
        prom_metric.set(value)
        data_metrics[f"evidently_{i}"] = prom_metric

    logging.info(
        "Evidently data metrics were translated to Prometheus for querying and \
            displaying them on Prometheus and Grafana."
    )

    return data_metrics


if __name__ == "__main__":

    metrics = monitor_evidently()
    app.run(host="0.0.0.0", port=9091)
