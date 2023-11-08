# Import libraries
from typing import Any

import mlflow
from mlflow.entities import ViewType
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient


def transition_to_stage(
    client: MlflowClient,
    model_name: str,
    model_version: str,
    new_stage: str,
    archive: bool,
) -> None:
    """
    Transitions a model to a defined stage
    """

    # Transition the model to the stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=archive,
    )

    print(f'Version {model_version} of the model {model_name} has been transitioned to {new_stage}')
    print('\n')


def load_models(model_name: str, stage: str) -> Any:
    """
    Loads the latest model saved before given the modele name and stage
    """
    # Get the mdoel in the stage
    model_stage_uri = f'models:/{model_name}/{stage}'
    print(f'Loading registered {stage} model version from URI: {model_stage_uri}')
    print('\n')

    model = mlflow.pyfunc.load_model(model_stage_uri)

    return model

def get_latest_version(client: MlflowClient, model_name: str, stage: str) -> str:
    """
    Finds the version number of the latest version of a model in a particualr stage.
    """
    # Get the information for the latest version of the model in a given stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_stage_version = latest_version_info[0].version

    print(f'The latest {stage} version of the model {model_name} is {latest_stage_version}')
    print('\n')

    return latest_stage_version


def delete_version(client: MlflowClient, model_name: str, model_version: str) -> None:
    """
    Deletes a specific version of a model permanently
    """
    client.delete_model_version(
        name=model_name,
        version=model_version
    )

    print(f'The version {model_version} of the model {model_name} has been permanently deleted')
    print('\n')


def update_model_version(
    client: MlflowClient, model_name:str, model_version: str, version_description: str
) -> str:
    """
    Adds a description to a version of the model
    """
    response = client.update_model_version(
        name=model_name,
        version=model_version,
        description=version_description
    )

    run_id = response.run_id

    print(f'Description has been added to the version {model_version} of the model {model_name}')
    print('\n')

    return run_id


def update_registerd_model(client: MlflowClient, model_name: str, description: str) -> str:
    """
    Adds a description to the model
    """
    response = client.update_registered_model(name=model_name, description=description)

    mod_name = response.name

    print(f'Description has beed added to the model {model_name}\n')

    return mod_name


def wait_until_ready(client: MlflowClient, model_name: str, model_version: str):
    """
    After creating a model version, it may take a short period of time to become ready.
    Certain operations, such as model stage transitions, require the model to be in the
    READY state. Other operations, such as adding a description or fetching model details,
    can be performed before the model version is ready (for example, while it is in the
    PENDING_REGISTRATION state).

    Uses the MlflowClient.get_model_version() function to wait until the model is ready.
    """
    status = 'Not ready'

    while status == "Not ready":
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version
        )

        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f'Model status: {ModelVersionStatus.to_string(status)}')
        print('\n')


def register_model(
    best_run_id: str,
    artifact_path: str,
    model_name: str,
) -> dict[str, Any]:
    """
    Register the model
    """

    # Register the model
    model_uri = f'runs:/{best_run_id}/{artifact_path}'
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    print('\n')
    print(f'Version {model_details.version} of the model {model_details.name} has been registered')
    print('\n')
    print(f'Model details: \n{model_details}\n')

    return model_details


def list_experiments(client: MlflowClient) -> list[dict]:
    """
    Gets all existing experiments
    """
    experiments = client.list_experiments()

    return experiments


def get_experiment_id(experiment_name: str) -> str:
    """
    Finds the experiment id of a specific experiment by its name
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    return experiment_id


def search_runs(client: MlflowClient, experiment_id: str, order_by: str, max_results: int = 10000) -> list[dict]:
    """
    Searches and brings all info of the runs belonging to a specific experiment which is introduced to the function by its id
    """

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=[order_by]
    )

    return runs

def get_best_params(client: MlflowClient, experiment_name: str, order_by: str, max_results: int = 1000) -> tuple[dict[str, Any], str]:
    """
    Get the parameters of the best model run of a particular experiment
    """
    # Get the id of the found experiment
    experiment_id = get_experiment_id(experiment_name)

    # Get the pandas data frame of the experimetn results in the ascending order by RMSE
    runs = search_runs(client, experiment_id, order_by, max_results)

    # Get the id of the best run
    best_run_id = runs[0].info.run_id

    # Get the best model parameters
    best_params = runs[0].data.params
    rmse = runs[0].data.metrics['rmse']
    
    print(f'\nBest parameters from the run {best_run_id} of {experiment_id}/{experiment_name}\n')

    print('rmse:', rmse)
    for key, value in best_params.items():
        print(f"{key}, {value}")
    print('\n')

    return best_params

