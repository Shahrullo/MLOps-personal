# Import libraries
from datetime import datetime
from typing import Any

import pytest
from airflow.models import DAG

from dags.monitoring.evidently_monitoring import (
    create_evidently_reports,
    monitor_evidently,
    prepare_evidently_data,
)


@pytest.fixture
def the_dag() -> Any:

    """
    Sets the dag as a fixture for repetitive use.
    """

    dag = DAG(dag_id="bike_sharing_trainer_v1", start_date=datetime(2022, 8, 25, 2))

    return dag


def test_prepare_evidently_data() -> None:

    """
    Tests loading, transformation and preparation of the training and
    validation feature datasets for creating evidently reports and metrics.
    """

    data_frame = prepare_evidently_data()

    assert data_frame.columns.to_list() == [
        'temp',
        'atemp',
        'humidity',
        'windspeed',
        'season',
        'holiday',
        'workingday',
        'weather',
        "target",
        "prediction",
    ]


def test_create_evidently_reports() -> None:

    """
    Tests creating dashboards.
    """

    dashboards = create_evidently_reports()

    assert dashboards == [
        "data_drift_dashboard",
        "data_target_drift_dashboard",
        "regression_model_performance_dashboard",
        "data_quality_dashboard",
    ]


def test_monitor_evidently() -> None:

    """
    Tests recording data drift evaluation results in MLFlow.
    """

    metrics = monitor_evidently(1)

    assert (
        str(metrics["evidently_1"])
        == "gauge:evidently:num_target_drift:count:reference"
    )
    assert (
        str(metrics["evidently_68"])
        == "gauge:evidently:regression_performance:feature_error_bias:total_bill_num_ref_over"
    )
    assert (
        str(metrics["evidently_195"])
        == "gauge:evidently:data_quality:quality_stat:reference_total_bill_num_most_common_value"
    )
    assert (
        str(metrics["evidently_278"])
        == "gauge:evidently:data_quality:quality_stat:current_size_num_mean"
    )
