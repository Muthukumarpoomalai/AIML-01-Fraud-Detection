# Real-Time Financial Fraud Detection Platform

## 1. Project Overview

A machine learning-based financial fraud detection platform designed to identify potentially fraudulent transactions using transaction data, machine learning, graph neural networks and streaming technologies.

The project combines model training, feature engineering, experiment tracking, transaction streaming, model serving, containerization, orchestration and data drift monitoring.

## 2. Dataset

**Dataset:** IEEE-CIS Fraud Detection  
**Source:** https://www.kaggle.com/competitions/ieee-fraud-detection/data

Dataset statistics:

- Total transactions: 590,540
- Legitimate transactions: 569,877
- Fraudulent transactions: 20,663
- Fraud percentage: approximately 3.50%

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine Learning | XGBoost |
| Graph Neural Network | PyTorch, PyTorch Geometric, GraphSAGE |
| Streaming | Apache Kafka |
| Feature Store | Feast |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Dashboard | Streamlit |
| Drift Monitoring | Evidently AI |
| Testing | Pytest |

## 4. Project Features

### Exploratory Data Analysis
- Transaction distribution analysis
- Fraud and legitimate transaction analysis
- Missing value analysis
- Fraud percentage analysis

### Feature Engineering
- Transactions in the previous hour
- Transactions in the previous 24 hours
- Transaction amount compared with card average
- Device change indicator
- Address change indicators

### Machine Learning
- XGBoost fraud classification
- GraphSAGE-based transaction relationship modelling
- Weighted XGBoost and GNN score blending

### Streaming
- Kafka producer publishes transaction events.
- Kafka consumer receives transaction events.
- Consumer forwards transactions to the FastAPI prediction endpoint.

### Feature Store
- Feast feature definitions
- Offline feature source
- SQLite online store
- Online feature retrieval

### Experiment Tracking
- MLflow experiment tracking
- Model parameters
- Evaluation metrics
- Model artifacts

### Model Serving
- FastAPI REST API
- Fraud probability prediction
- Fraud or legitimate classification

### Deployment
- Docker image
- Kubernetes deployment
- Kubernetes NodePort service

### Monitoring
- Evidently AI data drift report based on historical data batches.

### Dashboard
- Streamlit interface
- Transaction input
- Fraud probability
- Prediction result

## 5. Model Evaluation

XGBoost evaluation results:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.7802 |
| PR-AUC | 0.1424 |

The dataset contains approximately 3.50% fraudulent transactions, making class imbalance an important consideration.

ROC-AUC and PR-AUC are ranking metrics, not classification accuracy.

## 6. Project Structure

```text
AIML-01-Fraud-Detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── features/
│   ├── models/
│   ├── streaming/
│   ├── api/
│   └── monitoring/
│
├── dashboard/
├── tests/
├── configs/
├── docs/
├── feature_repo/
├── models/
├── reports/
├── mlruns/
├── k8s/
│
├── Dockerfile
├── requirements-api.txt
├── requirements-dashboard.txt
└── README.md
```

## 7. Running the Project

### Step 1: Install dependencies

Activate the Python virtual environment and install the required dependencies.

```powershell
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-dashboard.txt
```

Install the additional ML, streaming and monitoring dependencies from the project's development environment as required.

### Step 2: Run the API locally

```powershell
uvicorn src.api.main:app --reload
```

API:
http://127.0.0.1:8000

Swagger documentation:
http://127.0.0.1:8000/docs

### Step 3: Run the Kafka pipeline

Start Kafka using:

```powershell
docker compose -f src/streaming/docker-compose.yml up -d
```

Run the consumer and producer using separate terminals:

```powershell
python src/streaming/consumer.py
```

```powershell
python src/streaming/producer.py
```

### Step 4: Run the Streamlit dashboard

```powershell
python -m streamlit run dashboard/app.py
```

Dashboard:
http://localhost:8501

### Step 5: Deploy with Docker

Build the image:

```powershell
docker build -t fraud-detection-api:1.0 .
```

### Step 6: Deploy with Kubernetes

```powershell
kubectl apply -f k8s/deployment.yaml
```

Check deployment:

```powershell
kubectl get pods
kubectl get services
```

API:
http://localhost:30080

Swagger:
http://localhost:30080/docs

### Step 7: Run drift monitoring

```powershell
python src/monitoring/drift_monitor.py
```

Generated report:

`reports/data_drift_report.html`

### Step 8: Run tests

```powershell
python -m pytest -q
```

## 8. Implementation Notes and Limitations

- The deployed FastAPI service currently uses the XGBoost sample model.
- GraphSAGE training and weighted score blending are implemented as separate model workflows and artifacts; the blended model is not currently used by the live API.
- Feast online feature retrieval has been tested independently but is not connected to live API inference.
- Kafka-to-FastAPI transaction forwarding has been tested.
- Evidently AI currently compares historical data batches; continuous streaming drift monitoring has not been implemented.
- Graph relationships use available transaction attributes as proxies, not verified customer identities or confirmed fraud rings.
- The GNN experiment uses a limited dataset sample.

These limitations distinguish the implemented components from integrations that would be required for a production deployment.

## 9. Future Improvements

- Integrate Feast online features into API inference.
- Integrate and validate combined XGBoost and GNN predictions.
- Implement continuous drift monitoring.
- Improve graph construction using reliable customer or account identifiers.
- Add production-grade logging, security and monitoring.

## 10. Conclusion

This project demonstrates an end-to-end financial fraud detection workflow involving machine learning, graph neural networks, streaming, feature management, experiment tracking, API development, containerization, orchestration and monitoring.

The platform provides a foundation for further development into a production-ready fraud detection system.