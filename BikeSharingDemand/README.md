# Bike Sharing Demand Forecasting

This repository details an MLOps project where we utilize XGBoost to forecast bike-sharing demand based on several features including _datetime, season, holiday, workingday, wather,_ and others. We start by provisioning AWS resources using _Terraform_. After setting up our data, model, and scripts locally and passing all tests, we push our project directory to _GitHub_. This triggers _GitHub Actions_ to install our setup onto an Ubuntu virtual machine.

Following setup, we launch _MLflow_ and _Airflow_ servers for experiment tracking and workflow management, respectively. _Airflow_ is configured to run monthly, automatically detecting and importing new data from the _S3_ bucket to retrain the model. The updated model is then stored in the _S3_ bucket for production use. We monitor the performance metrics through _Evidently_ locally and on the _Grafana_ dashboard via _Prometheus_. Alerts for data and concept drift, along with model retraining notifications, are sent out via email.

The service is accessible through a _Flask_-based web interface.

Before committing changes to _GitHub_, we conduct local testing using _pytest, black, isort, localstack, pre-commit,_ and _pylint_.

<br>

## Installation

To set up the project, follow these steps:

1. Sign up for an AWS account and configure programmatic access.

2. To deploy AWS resources with _Terraform_, navigate to the terraform directory and execute:

```terraform
terraform init
terraform plan
terraform apply -auto-approve
```

3. Push your project to GitHub:

```bash
git add .
git commit -m 'initial commit'
git push origin main
```

Use `wget` if _GitHub Actions_ are not configured.

4. To install dependencies like _brew_, initialize _Prometheus_ and _Grafana_ servers, and set up _PostgreSQL_ databases on _RDS_, run the following in the project directory:

```bash
./start.sh
```
<br>

## Usage

To operate the application, carry out these steps:

1. Boot up the MLflow server with:

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://s3b-bike-sharing-demand/mlflow/
```

2. Start the Evidently server:

```bash
mlflow server -h 0.0.0.0 -p 5500 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/DB_NAME --default-artifact-root s3://s3b-bike-sharing-demand/evidently/
```

Ensure you have your database credentials and RDS endpoint handy.

3. Modify `airflow.cfg` to set `LocalExecutor` as the executor and configure the `sql_alchemy_conn` string with your database connection details.

4. In the `/app/Bike-Sharing-Demand` project folder, initialize the _Airflow_ database:

```bash
airflow db init
```

5. Create an _Airflow_ user:

```bash
airflow users create --username <username> --password <password> --firstname <firstname> --lastname <lastname> --role Admin --email <email>
```

6. Launch the _Airflow_ webserver:

```bash
airflow webserver -p 8080 -D
```

7. Start the _Airflow_ scheduler:

```bash
airflow scheduler -D
```

8. Configure port forwarding in Visual Studio Code for the following ports to access the application's web interfaces:

* 3000: _Grafana_
* 3500: _Flask_ (Prediction Service)
* 3600: _Evidently_ Reports
* 5000: _MLflow_
* 8080: _Airflow_ Web Server
* 8793: _Airflow_ Scheduler
* 9090: _Prometheus_
* 9091: _Prometheus_ (Application Metrics)

You can streamline the starting of _MLflow_ and _Airflow_ servers using the `Makefile`, ensuring the MLflow server endpoints are correctly specified:

```bash
make mlflow_5000
make mlflow_5500
make airflow_web
make airflow_scheduler
```