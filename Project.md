# **MLOps Case Study, Building a Real-Time Predictive System (RPS)**

# **Deadline Nov 30, 2025**

**Context:** You are working as an MLOps team for a forward-thinking tech company that needs to move beyond static, periodically trained machine learning models. Your goal is to develop a robust, automated, and continuously monitored **Real-Time Predictive System (RPS)** that handles live data streams and automatically adapts to changing patterns (**concept drift**).

The deliverable is a fully integrated MLOps pipeline that not only trains a model but also serves and monitors it in a production-like environment.

#### **1. Phase I: Problem Definition and Data Ingestion**

The first step is selecting a real-world problem and establishing a reliable, automated data feed.

#### **The Challenge (Step 1)**

Each group must select *one* predictive challenge centered around **time-series data** from a **free, live external API**.

| Domain        | Example<br>Free<br>APIs             | Predictive<br>Task<br>(Goal)                                                                                                                                     |
|---------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Financial     | Alpha<br>Vantage,<br>Twelve<br>Data | Stock/Crypto<br>Volatility:<br>Predict<br>the<br>short-term<br>volatility<br>or<br>closing<br>price<br>for<br>a<br>specific<br>asset<br>(e.g.,<br>next<br>hour). |
| Environmental | OpenWeatherMap                      | Localized<br>Forecasting:<br>Predict<br>a<br>key<br>weather<br>variable<br>(e.g.,<br>temperature,<br>wind<br>speed)<br>for<br>a<br>city<br>4-6<br>hours<br>out.  |

| Logistics | Public<br>Transit<br>Data | ETA/Delay:<br>Predict<br>the<br>delay<br>or<br>arrival       |
|-----------|---------------------------|--------------------------------------------------------------|
|           |                           | time<br>of<br>a<br>specific<br>transit<br>vehicle<br>at<br>a |
|           |                           | future<br>stop.                                              |
|           |                           |                                                              |
|           |                           |                                                              |

## **The Orchestration Core: Apache Airflow (Step 2)**

Your MLOps pipeline will be structured as a **Directed Acyclic Graph (DAG)** in **Apache Airflow**. This DAG must run on a schedule (e.g., daily) and be responsible for the entire ETL and model retraining lifecycle.

- **Extraction (2.1):** Use a Python operator to connect to your chosen API and fetch the latest live data. The raw data must be saved immediately, stamped with the collection time.
  - **Mandatory Quality Gate:** Implement a strict **Data Quality Check** right after extraction (e.g., check for >1% null values in key columns, or schema validation). If the data quality check fails, the DAG *must* fail and stop the process.
- **Transformation (2.2):** Clean the raw data and perform essential **feature engineering** specific to your time-series problem (e.g., creating lag features, rolling means, or time-of-day encodings).
  - **Documentation Artifact:** Use **Pandas Profiling** or a similar tool to generate a detailed data quality and feature summary report.<sup>1</sup> This report *must* be logged as an artifact to your MLflow Tracking Server (Dagshub).
- **Loading & Versioning (2.3 & 3):**
  - The final, processed dataset must be stored in a cloud-like object storage (e.g., **MinIO**, **AWS S3**, or **Azure Blob Storage**).
  - **Data Version Control (DVC):** Use **DVC** to version this processed dataset. The small .dvc metadata file will be committed to Git, but the large dataset file itself must be pushed to your chosen remote storage.

### **2. Phase II: Experimentation and Model Management**

#### **MLflow & Dagshub Integration (Step 4)**

The Airflow DAG will trigger the model training script (train.py). This is where robust experimentation and artifact management come into play.

- **MLflow Tracking:** Use **MLflow** within the training script to track every experiment run. Log all **hyperparameters**, key **metrics** (e.g., RMSE, MAE, R-squared), and the final **trained model** as an artifact.
- **Dagshub as Central Hub:** Configure **Dagshub** to act as your remote **MLflow Tracking Server** and **DVC remote storage**. This ensures all three core components—**Code** (Git), **Data** (DVC), and **Models/Experiments** (MLflow)—are linked and visible in a single, collaborative UI.

### **3. Phase III: Continuous Integration & Deployment (CI/CD)**

The code lifecycle must be rigorous, using a professional branching strategy and automated checks.

### **Git Workflow and CI Pipeline (Step 5)**

- **Strict Branching Model (5.1):** Adhere strictly to the **dev**, **test**, and **master** branch model. All new work begins on feature branches and merges into dev.
- **Mandatory PR Approvals (5.3):** Enforce **Pull Request (PR) approval** from at least one peer before merging into the test and master branches.

## **GitHub Actions with CML (5.1 & 5.2)**

Use **GitHub Actions** to automate the CI/CD pipeline, and integrate **CML (Continuous Machine Learning)** for automated reporting.

| Merge<br>Event                    | CI<br>Action                                                                                                                                | CML<br>Integration                                                                                                                                                                                                                                                                                                                                                                     |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Feature<br>\$\rightarrow\$<br>dev | Run<br>Code<br>Quality<br>Checks<br>(Linting)<br>and<br>Unit<br>Tests.                                                                      | N/A                                                                                                                                                                                                                                                                                                                                                                                    |
| dev<br>\$\rightarrow\$<br>test    | Model<br>Retraining<br>Test:<br>Trigger<br>the<br>Airflow<br>DAG<br>to<br>run<br>a<br>full<br>data<br>and<br>model<br>training<br>pipeline. | Use<br>CML<br>to<br>automatically<br>generate<br>and<br>post<br>a<br>metric<br>comparison<br>report<br>in<br>the<br>Pull<br>Request<br>comments,<br>comparing<br>the<br>newly<br>trained<br>model's<br>performance<br>against<br>the<br>existing<br>production<br>model<br>in<br>master.<br>The<br>merge<br>should<br>be<br>blocked<br>if<br>the<br>new<br>model<br>performs<br>worse. |
| test<br>\$\rightarrow\$<br>master | Full<br>Production<br>Deployment<br>Pipeline.                                                                                               | N/A                                                                                                                                                                                                                                                                                                                                                                                    |

# **Containerization and Deployment (5.4 & 5.5)**

- **Docker Containerization (5.4):** The model must be served via a **REST API** (using **FastAPI** or **Flask**) inside a **Docker container**. This is your deployable unit.
- **Continuous Delivery:** The **test \$\rightarrow\$ master** merge must trigger the final CD steps:

- 1. Fetch the best-performing model from the MLflow **Model Registry**.
- 2. **Build** the final Docker image.
- 3. **Push** the tagged image (e.g., app:v1.0.0) to a container registry (e.g., **Docker Hub**).<sup>2</sup>
- 4. **Deployment Verification:** Run the container on a minimal host (e.g., a simple docker run) to verify the service starts and responds correctly to a health check.

# **4. Phase IV: Monitoring and Observability**

A production system is incomplete without monitoring. You must implement tools to ensure the model remains reliable.

## **Prometheus and Grafana**

- **Prometheus:** Embed a **Prometheus data collector** within your FastAPI prediction server. This service must expose metrics endpoints.
  - **Service Metrics:** Collect API inference **latency** and total **request count**.
  - **Model/Data Drift Metrics:** Expose crucial custom metrics, such as the **ratio of prediction requests containing out-of-distribution feature values** (a basic data drift proxy).
- **Grafana:** Deploy **Grafana** and connect it to your Prometheus instance.
  - **Live Dashboard:** Create a dashboard to visualize the key service and model health metrics in real time.
  - **Alerting Enhancement:** Configure a **Grafana Alert** to fire (e.g., log to a Slack channel or file) if the **inference latency** exceeds an acceptable threshold (e.g., 500ms) or if the **data drift ratio** spikes.

### **Summary of Tools to Integrate**

| Category           | Tools                                | Purpose<br>in<br>this<br>Project                                                                          |
|--------------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Orchestration      | Airflow                              | Schedule<br>and<br>automate<br>the<br>entire<br>ETL<br>\$\rightarrow\$<br>Training<br>workflow.           |
| Data/Model<br>Mgmt | DVC,<br>MLflow,<br>Dagshub           | Version<br>data,<br>track<br>experiments,<br>and<br>centralize<br>code,<br>data,<br>and<br>models.        |
| CI/CD              | GitHub<br>Actions,<br>CML,<br>Docker | Automate<br>testing,<br>model<br>comparison,<br>image<br>building,<br>and<br>deployment.                  |
| Monitoring         | Prometheus,<br>Grafana               | Collect<br>service/model<br>metrics,<br>visualize<br>performance,<br>and<br>alert<br>on<br>drift/latency. |

The success of your project will be measured by the seamless **automation and integration** of these tools, proving your ability to manage an ML model across its entire lifecycle.