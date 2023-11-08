# Import libraries
import sys

sys.path.insert(1, "/home/ubuntu/app/Bike-Sharing-Demand/dags/utils")

from typing import Any, Union
import mlflow
import numpy as np

import aws_utils as aws


def load_model() -> Any:
    """
    Loads the best model from the local fodler or cloud bucket.
    """

    logged_model = aws.get_parameter("logged_model")
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model


def predict(input_data: dict[str, Union[int, float]]) -> float:
    
    """
    Using the best model and features entered by user,
    predicts the target value.
    """

    # Load the model
    model = load_model()

    # Make the prediction
    prediction = model.predict(input_data)[0]

    return f'{prediction:2f}'