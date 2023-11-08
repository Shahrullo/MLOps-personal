#!/bin/bash
sudo apt update -y
cd /home/ubuntu
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt intsall python3.9 -y
alias python3=python3.9
sudo apt install awscli -y
mkdir app
cd ./app
mkdir Bike-Sharing-Demand
cd ..
sudo chmod -R 707 /home/ubuntu/app
echo 'export PATH="${HOME}/app/Bike-Sharing-Demand:${PATH}"' >> .bashrc
echo 'alias python3=python3.9' >> .bashrc
echo 'export PGPASSWORD=password' >> .bashrc
source .bashrc
sudo apt install python3-pip -y
aws s3 cp s3://s3b-bike-sharing/config/requirements.txt ./app/Bike-Sharing-Demand
aws s3 cp s3://s3b-bike-sharing/config/prometheus-config.yml ./app/Bike-Sharing-Demand
aws s3 cp s3://s3b-bike-sharing/config/start.sh ./app/Bike-Sharing-Demand
chmod +x ./app/Bike-Sharing-Demand/start.sh
cd ./app/Bike-Sharing-Demand
python3.9 -m pip install flask
pip install -r requirements.txt
sudo apt-get install build-essential procps curl file git -y
sudo apt-get install -y apt-transport-https
sudo apt-get install -y software-properties-common wget
sudo wget -q -O /usr/share/keyrings/grafana.key https://packages.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://packages.grafana.com/oss/deb stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install grafana -y
sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server.service
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql.service
cd ../../
echo 'export AIRFLOW_HOME=/home/ubuntu/app/Bike-Sharing-Demand' >> .bashrc
source .bashrc
