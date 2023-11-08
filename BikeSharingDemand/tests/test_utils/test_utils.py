# Import libraries
import boto3
from mlflow.tracking import MlflowClient

from dags.utils.airflow_utils import get_vars
from dags.utils.aws_utils import (
    create_parameter,
    get_bucket_object,
    get_parameter,
    put_object,
    send_sns_topic_message,
)
from dags.utils.mlflow_utils import (
    get_latest_version,
    list_experiments,
    search_runs,
    update_model_version,
    update_registered_model,
)

mlflow_client = MlflowClient("http://127.0.0.1:5000")


def test_get_vars() -> None:

    """
    Tests all variables created and stored on
    Airflow Variables.
    """

    # Retrieve all variables stored in Airflow
    (
        bucket,
        file_name,
        local_path,
        local_data_transformed_filename,
        mlflow_experiment_name,
        evidently_experiment_name,
        model_name,
    ) = get_vars()

    assert bucket == "s3b-bike-sharing-demand"
    assert file_name == "data/train.csv"
    assert local_path == "/home/ubuntu/app/Bike-Sharing-Demand/"
    assert (
        local_data_transformed_filename
        == "/home/ubuntu/app/Bike-Sharing-Demand/data/bike_transformed.csv"
    )
    assert mlflow_experiment_name == "mlflow-experiment-1"
    assert evidently_experiment_name == "evidently-experiment-1"
    assert model_name == "xgboost-model"


def test_get_parameter() -> None:

    """
    Tests getting a parameter from AWS Parameter Store.
    """

    # Retrieve an existing parameter
    parameter = get_parameter("sns_topic_arn")

    assert parameter == "bshd:aws:sns:ap-northeast-2:830193445063:BikeSharingDemandTopic"

def test_create_parameter() -> None:

    """
    Tests putting a parameter in AWS Parameter Store.
    """

    # Update an existing parameter, version number is going to be the next one.
    version = create_parameter(
        "model_name",
        "Name of the XGBoost model as a test",
        "xgboost-bike-sharing",
        "String",
    )

    assert isinstance(version, int) is True
    assert version == 7


def test_get_bucket_object() -> None:

    """
    Tests getting an object from the S3 bucket.
    """

    # Get the etag of an existing object in the S3 bucket
    _, object_etag = get_bucket_object("s3b-bike-sharing-demand", "data/current.csv")
    object_etag = object_etag.replace('"', "")

    assert isinstance(object_etag, str) is True
    assert object_etag == "fe6556df2c2fc42bd8bc3dc247fce0d9"


def test_put_object() -> None:

    """
    Tests putting an existing object in the S3 bucket.
    """

    bucket = "s3b-bike-sharing-demand"
    key = "data/train.csv"

    # Upload a dummy object into the S3 bucket
    put_object(
        "/home/ubuntu/app/Bike-Sharing-Demand/data/train.csv",
        bucket,
        key,
        "Name",
        "data",
    )

    # Get the tag of the uploaded object to verify.
    # It is correct if the object is uploaded with the given tag successfully
    response = boto3.client("s3", region_name="eu-west-1").get_object_tagging(
        Bucket=bucket, Key=key
    )

    tag_key = response["TagSet"][0]["Key"]
    tag_value = response["TagSet"][0]["Value"]

    assert tag_key == "Name"
    assert tag_value == "data"


def test_send_sns_topic_message() -> None:

    """
    Tests publishing a message to SNS topic.
    """

    # Create a message to send to the SNS topic
    topic_arn = "bshd:aws:sns:ap-northeast-2:830193445063:BikeSharingDemandTopic"
    subject = "Test for SNS message publishing"
    message = "This is a message that shows the successful completion of the test of SNS message publishing."
    message_id = send_sns_topic_message(topic_arn, message, subject)

    assert isinstance(message_id, str) is True


def test_get_latest_version() -> None:

    """
    Tests getting the latest version of the model in production stage in MLFlow.
    """

    # Find the latest version in MLFlow
    latest_stage_version = get_latest_version(
        mlflow_client, "xgboost-model", "Production"
    )

    assert latest_stage_version == "2"


def test_update_model_version() -> None:

    """
    Tests updating a model version's description.
    """

    version_description = "This is a test description"

    run_id = update_model_version(
        mlflow_client, "xgboost-model", "4", version_description
    )

    assert isinstance(run_id, str) is True
    assert run_id == "7c64eed9f6814438a170d4537b4bd713"


def test_update_registered_model() -> None:

    """
    Tests updating the registered model's description.
    """

    description = """
      This model predicts the bike sharing for serving bike rental.
      Bike sharing data consists of ten features
      """

    # Returns the name of the registered model updated
    mod_name = update_registered_model(mlflow_client, "xgboost-model", description)

    assert isinstance(mod_name, str) is True
    assert mod_name == "xgboost-model"


def test_list_experiments() -> None:

    """
    Tests getting the list of experiments.
    """

    # List of all experiments with full information
    experiments = list_experiments(mlflow_client)

    # List of experiments by name only
    experiment_names = [experiment.name for experiment in experiments]

    assert "mlflow-experiment-1" in experiment_names


def test_search_runs() -> None:

    """
    Tests searching for info of all runs belonging to a specific
    experiment which is introduced to the function by its id.
    """

    runs = search_runs(mlflow_client, "1", "metrics.rmse ASC", 5000)

    # The name of the best model (the top one)
    model = runs[0].data.tags["model"]

    assert model == "xgboost"
