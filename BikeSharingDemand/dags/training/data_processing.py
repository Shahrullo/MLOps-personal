# Import libraries
import json
import logging
import pickle
from datetime import datetime
from typing import Any, Literal, Union

import mlflow
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.tracking import MlflowClient
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from ..utils.airflow_utils import get_vars
from ..utils.aws_utils import create_parameter, get_parameters, send_sns_topic_message
from ..utils.mlflow_utils import (
    delete_version,
    get_best_params,
    get_latest_version,
    list_experiments,
    load_models,
    register_model,
    search_runs,
    transition_to_stage,
    update_model_version,
    update_registerd_model,
    wait_until_ready
)

# We store variables that won't cahgne often in AWS Parameter Store
tracking_server_host = get_parameters("tracking_server_host") # This can be local: 127.0.0.1 or EC2, e.g.: ex2-54-75-6-9.ap-northeast-2.compute.amazonaws.com

# Set the tracking server uri
MLFLOW_PORT = 5000
mlflow_tracking_uri = f'http://{tracking_server_host}:{MLFLOW_PORT}'
mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow_client = MlflowClient(mlflow_tracking_uri)

# Retrieve the inital path and mlflow artifact path from AWS Parameter Store
try:
    mlflow_artifact_path = json.loads(get_parameters("artifact_paths"))["mlflow_model_artifacts_path"] # model_mlflow
    mlflow_initial_path = json.loads(get_parameters("initial_paths"))["mlflow_model_initial_path"] # s3://s3b-bike-sharing-demand/mlflow/
except:
    mlflow_artifact_path = "models_mlflow"
    mlflow_initial_path = "s3://s3b-bike-sharing-demand/mlflow/"

def search_best_parameters(tag: str) -> dict[str, Union[int, float]]:
    """
    Searches and finds the optimum parameters within the defined ranges with Hyperopt.
    Logs them in MLflow.
    """
    
    # Rerieve variables
    _, _, local_path, _, experiment_name, _, _ = get_vars()

    # Load data from the local disk
    x_train = loadtxt(f'{local_path}data/X_train.csv', delimiter=',')
    x_val = loadtxt(f'{local_path}data/X_val.csv', delimiter=',')
    y_train = loadtxt(f'{local_path}data/y_train.csv', delimiter=',')
    y_val = loadtxt(f'{local_path}data/y_val.csv', delimiter=',')
    logging.info("Training and validation datasets are retrieved from the local storage")

    # Convert to DMatrix data structure for XGBoost
    train = xgb.DMatrix(x_train, label=y_train)
    valid = xgb.DMatrix(x_val, label=y_val)
    logging.info("Training and validation matrix datasets are created for XGBoost")

    # Set an mlflow experiment
    mlflow.set_experiment(experiment_name)
    logging.info(f"Tracking server host {tracking_server_host} is retrieved from AWS Parameter Store")
    logging.info(f"Tracking uri {mlflow_tracking_uri} is set in mlflow.")
    logging.info(f"Check the tracking uri: {mlflow.get_tracking_uri()}")
    logging.info(f"MLFlow experiment {experiment_name} is set.")

    # Search for best parameters
    def objective(params: dict) -> dict[str, Union[float, Literal['ok']]]:
        with mlflow.start_run():
            mlflow.set_tag("model", tag)
            mlflow.log_param(params)

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, "validation")],
                early_stopping_rounds=50,
            )

            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        logging.info(f"Loss: {rmse} and status: {STATUS_OK}")

        return {"loss": rmse, "status": STATUS_OK}

    # Search space for parameters
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "colsample_bytree": hp.choice("colsample_bytree", np.arange(0.3, 0.8, 0.1)),
        "subsample": hp.uniform("subsample", 0.8, 1),
        "n_estimators": 100,
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    best_results = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )


def find_best_params(ti: Any, metric: str, max_results: int) -> dict[str, Any]:
    """
    Retrieves the parameters of the best model run of a particular experiment from the mlflow utils module
    """
    # Retrieve variables
    _, _, _, _, experiment_name, _, _ = get_vars()

    # Get the best params from mlflow server
    best_params = get_best_params(mlflow_client, experiment_name, metric, max_results)
    logging.info(f'The resulting bets parameters of the experiment {experiment_name} by the metric {metric} are {best_params}')

    # Push the best params to XCom
    ti.xcom_push(key='best_params', value=best_params)
    logging.info(f'The resulting best parameters {best_params} of the experiment {experiment_name} are pushed to XCom')

    return best_params


def run_best_model(ti: Any, tag: str) -> str:
    """
    Runs the model with the best parameters searched and found
    at the earlier phase. Then, saves the model and info in the artifacts folder and bucket.
    """
    # Rerieve variables
    _, _, local_path, _, experiment_name, _, model_name = get_vars()

    best_params = ti.xcom_pull(key="best_params", task_ids=["find_best_params"])
    best_params = best_params[0]
    logging.info(f"Best params {best_params} are retrieved from XCom.")
    logging.info(f'MLflow artifact path {mlflow_artifact_path} is retrieved from AWS Parameter Store')

    # Load data from the local disk
    x_train = loadtxt(f"{local_path}data/X_train.csv", delimiter=",")
    x_val = loadtxt(f"{local_path}data/X_val.csv", delimiter=",")
    y_train = loadtxt(f"{local_path}data/y_train.csv", delimiter=",")
    y_val = loadtxt(f"{local_path}data/y_val.csv", delimiter=",")
    logging.info("Training and validation datasets are retrieved from the local storage")

    # Convert to DMatrix data structure to XGBoost
    train = xgb.DMatrix(x_train, label=y_train)
    valid = xgb.DMatrix(x_val, label=y_val)
    logging.info("Training and validation matrix datasets are created for XGBoost")

    # Set an mlflow experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        # Get the run_id of the best model
        best_run_id = run.info.run_id
        logging.info(f'Best run id: {best_run_id}')

        mlflow.set_tag("model", tag)
        mlflow.log_paras(best_params)

        # Train the XGBoost model with the best parameters
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Save the model (xgboost_model.bin) locally in the folder "../models/" (in case we want)
        with open(f"{local_path}models/xgboost_model.bin", "wb") as f_out:
            pickle.dump(booster, f_out)
        logging.info(f"XGboost modele is saved on the path '{local_path}models/xgboost_model.bin' of the local machine")

        # Save the model using 'log_artifact' in the defined artifacts
        mlflow.log_artifact(
            local_path=f'{local_path}models/xgboost_model.bin',
            artifact_path=mlflow_artifact_path
        )
        logging.info(f"Artifacts are saved on the artifact path {mlflow_artifact_path}.")

        # Save the model (booster) using 'log_model' in the defined artifacts folders/bucket
        # This is defined on the CLI and as artifacts path parameter on AWS Parameter Store:
        # s3://s3b-bike-sharing-demand/mlflow/ ... /models_mlflow/
        mlflow.xgboost.log_model(booster, artifact_path=mlflow_artifact_path)
        logging.info(f'XGBoost model is saved on the artifact path {mlflow_artifact_path}.')
        logging.info(f'Default artifacts URI: {mlflow.get_artifact_uri()}')

        # Push the best run id to XCom
        ti.xcom_push(key='best_run_id', value=best_run_id)
        logging.info(f'The best run id {best_run_id} of the model {model_name} is pushed to XCom.')


def register_best_model(ti: Any, version_description: str) -> dict[str, Any]:
    """
    Register the best run id, adss a high-level and version
    descriptions, and transitions the model to 'staging'
    """

    # Retrieve variables
    _, _, _, _, _, _, model_name = get_vars()

    # Get the best run id
    best_run_id = ti.xcom_pull(key='best_run_id', task_ids=['run_best_model'])
    best_run_id = best_run_id[0]
    logging.info(f'Best run id {best_run_id} is retrieved from XCom.')

    # Register the best model
    model_details = register_model(best_run_id, mlflow_artifact_path, model_name)
    logging.info(f'Model {model_name} with the run id {best_run_id} is registered on artifact path {mlflow_artifact_path}')

    # Wait until the model is ready
    wait_until_ready(mlflow_client, model_details.name, model_details.version)
    logging.inf(f'Modele {model_name} is ready for further processing')

    # Add a high-level description to the registered model,
    # including the machine learning problem and dataset
    description = """
        This model predicts the bike sharing demand given to the input features.
        Bike Sharing Demand data consists of 10 features given in the Kaggle platform
        """
    update_registerd_model(mlflow_client, model_details.name, description)
    logging.info(f'A high-level description {description} is added to the model {model_name}.')

    # Add a model version description with information about the model artchitecture and 
    # machine leanring framework
    update_model_version(mlflow_client, model_details.name, model_details.version, version_description)
    logging.info(f"A version description {version_description} is added to the model {model_name}.")

    # Transition the model to Staging
    transition_to_stage(mlflow_client, model_details.name, model_details.version, "staging", False)
    logging.info(f'Model {model_name} is transitioned to "staging"')

    # Push model details to XCom
    model_details_dict = {}
    model_details_dict['model_name'] = model_details.name
    model_details_dict['model_version'] = model_details.version
    ti.xcom_push(key='model_details', value=model_details_dict)
    logging.info(f'Molde details {model_details_dict} of the registered model {model_name} is pushed to XCom.')


def test_model(ti: Any) -> dict[str, float]:
    """
    Calculates RMSE for each of the new model that is developed
    and transitioned to 'staging', and the previous model that is 
    already in the 'production' stage. Allows for comparison of both models.
    """
    # Retrieve variables
    _, _, local_path, _, _, _, model_name = get_vars()

    # Get model details from XCom
    model_details_dict = ti.xcom_pull(
        key="model_details", task_ids=['register_best_model']
    )
    model_details_dict = model_details_dict[0]
    model_name = model_details_dict['model_name']
    logging.info(f'Model details {model_details_dict} are retrieved from XCom.')

    # Load data from the local disk
    x_test = loadtxt(f'{local_path}data/X_val.csv', delimiter=',')
    y_test = loadtxt(f'{local_path}data/y_val.csv', delimiter=',')
    logging.info('Validation datasets are retrieved from the local storage')

    # Load the staging model, predict with the new data and calculate its RMSE
    model_staging = load_models(model_name, "staging")
    model_staging_predictions = model_staging.predict(x_test)
    model_staging_rmse = mean_squared_error(
        y_test, model_staging_predictions, squared=False
    )
    logging.info(f"model_staging_rmse: {model_staging_rmse}")

    try:
        # Load the production model, predict with the new data and calucate its RMSE
        model_production = load_models(model_name, 'production')
        model_production_predictions = model_production.predict(x_test)
        model_production_rmse = mean_squared_error(
            y_test, model_production_predictions, squared=False
        )
        logging.info(f'model_production_rmse: {model_production_rmse}')

    except Exception:
        print("It seems that there is not any model in the production stage yet. \
              Then, we transition the current model to production.")
        model_production_rmse = None

    rmse_dict = {}
    rmse_dict['model_production_rmse'] = model_production_rmse
    rmse_dict['model_staging_rmse'] = model_staging_rmse

    # Push the RMSEs of the staging and production models to XCom
    ti.xcom_push(key='rmses', value=rmse_dict)
    logging.inf(f'RMSEs of the staging and production models {rmse_dict} are pushed to XCom.')


def compare_models(ti: Any) -> None:
    """
    Based on the RMSEs of the models, compares which one yields less
    loss value. The one with lower RMSE will be in 'production' stage.
    The other with higher RMSE will either stay in 'staging' (if new model)
    or be transitioned to 'archive' (if previous model) after which an
    optional deletion is to be offered
    """
    # Get rmse_dict from XCom
    rmse_dict = ti.xcom_pull(key='rmses', task_ids=['test_model'])
    rmse_dict = rmse_dict[0]
    model_production_rmse = rmse_dict['model_production_rmse']
    model_staging_rmse = rmse_dict['model_staging_rmse']
    logging.info(f'RMSEs of the staging and production models {rmse_dict} are retrieved from XCom.')

    # Get model details from XCom
    model_details_dict = ti.xcom_pull(key='model_details', task_ids=['register_best_model'])
    model_details_dict = model_details_dict[0]
    model_name = model_details_dict['model_name']
    model_version = model_details_dict['model_versions']
    logging.info(f'Model details {model_details_dict} are retrieved from XCom')

    # Compare RMSEs
    # If there is a model in production stage already
    if model_production_rmse:
        # If the staging model's RMSE is lower than or equal to the production mdoels'
        # RMSE, transition the former model to production stage, and delete the previous
        # production model
        if model_staging_rmse <= model_production_rmse:
            transition_to_stage(
                mlflow_client, model_name, model_version, 'production', True
            )
            latest_stage_version = get_latest_version(
                mlflow_client, model_name, 'archived'
            )
            delete_version(mlflow_client, model_name, latest_stage_version)
    else:
        # If there is not any model in production stage already, transition the staging
        # model to production
        transition_to_stage(
            mlflow_client, model_name, model_version, "production", False
        )


def get_experiments_by_id() -> list[str]:
    """
    Lists all existing experimetnts and finds their ids.
    """
    # List of existing experiments
    experiments = list_experiments(mlflow_client)
    logging.info(f'Experiments {experiments} are retrieved from MLFlow')
    experiment_ids = [experiments.experiment_id for experiment in experiments]

    # Remove default experiment as it has no runs
    experiment_ids.remove("0")

    return experiment_ids


def get_top_run(metric: str) -> dict[str, Union[str, float]]:
    """
    Checks all runs in all experiments and finds the
    best run that returns the lowest loss value.
    """
    best_rmse_dict = {}

    # A very high error score as the baseline
    best_run_score = float('inf')

    # List of existing experiment ids
    
    experiment_ids = get_experiments_by_id()
    logging.info("Experiments by ids '%s' are available.", experiment_ids)

    for experiment_id in experiment_ids:

        try:
            # Best run of a particular experiment
            runs = search_runs(mlflow_client, experiment_id, metric, 10000)
            best_run_id = runs[0].info.run_id
            best_run_rmse = runs[0].data.metrics["rmse"]
            logging.info(
                "Experiment by id '%s' has the best run by id '%s' with the rmse score %s.",
                experiment_id,
                best_run_id,
                best_run_rmse,
            )

            # If the experiment has the best score, we store it.
            if best_run_rmse <= best_run_score:
                best_run_score = best_run_rmse
                best_rmse_dict["experiment_id"] = experiment_id
                best_rmse_dict["best_run_id"] = best_run_id
                best_rmse_dict["best_run_rmse"] = best_run_rmse
                logging.info(
                    "As of now, the run by id '%s' of the experiment by id '%s' has the \
                        best historical rmse score %s among all experiments' runs.",
                    best_run_id,
                    experiment_id,
                    best_run_rmse,
                )
        except Exception:
            print("Experiment by id '%s' has no runs at all.", experiment_id)

    # Build the path to the best model
    experiment_id = best_rmse_dict["experiment_id"]
    best_run_id = best_rmse_dict["best_run_id"]
    logged_model = f"{mlflow_initial_path}{experiment_id}/{best_run_id}/artifacts/{mlflow_artifact_path}/"
    logging.info(
        "The best historical model is given by the run '%s' of the experiment '%s'.",
        best_run_id,
        experiment_id,
    )

    # Create a parameter on AWS from the best model info to enable the prediction script to
    # use it later
    create_parameter("logged_model", "Path to the best model", logged_model, "String")
    logging.info("The best historical model is logged to '%s'.", logged_model)

    # Set the current date and time
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d / %H-%M-%S")
    logging.info("Date/time '%s' is set.", date_time)

    # Send an email about the final result.
    topic_arn = get_parameters("sns_topic_arn")
    subject = "Bike sharing demand prediction training results "
    message = f"As a result of the latest training performed on '{date_time}',\n\nthe best historical model is provided by the run '{best_run_id}' of the experiment '{experiment_id}',\n\nand it has a RMSE of {best_rmse_dict['best_run_rmse']}."
    send_sns_topic_message(topic_arn, message, subject)
    logging.info(
        "Email about the training results are sent to AWS SNS topic 'BikeSharingDemandTopic'."
    )

    return best_rmse_dict