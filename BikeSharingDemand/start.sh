#!/bin/bash

# Navigate to the project directory
cd /home/ubuntu/app/Bike-Sharing-Demand

# Install essential packages
sudo apt-get update
sudo apt-get install -y build-essential

# Install required tools and services
sudo apt-get install -y gcc prometheus postgresql

# Restart Prometheus service
sudo systemctl restart prometheus

# Copy Prometheus configuration file
sudo cp -f prometheus-config.yml /etc/prometheus/prometheus.yml

# Restart Prometheus service after copying configuration
sudo systemctl restart prometheus

# Display information about the Prometheus service
sudo systemctl status prometheus --no-pager

# Display information about the PostgreSQL service
sudo systemctl status postgresql --no-pager

# Display information about the Grafana service
sudo systemctl status grafana-server --no-pager

# Install Python dependencies
pip install chardet==4.0.0
pip install requests -U
pip install -U click
pip uninstall Flask-WTF -y
pip uninstall WTForms -y
pip install Flask-WTF==0.15.1
pip install WTForms==2.3.3

# Set AIRFLOW_HOME environment variable
export AIRFLOW_HOME=/home/ubuntu/app/Bike-Sharing-Demand

# Switch to the postgres user and create databases
sudo -u postgres psql \
   --host=bshd-rds-instance.cmpdlb9srhwd.ap-northeast-2.rds.amazonaws.com \
   --port=5432 \
   --username=postgres \
   --password \
   --dbname=mlflow_db <<EOF
CREATE DATABASE mlflow;
CREATE DATABASE airflow;
CREATE DATABASE evidently;
\l
EOF
