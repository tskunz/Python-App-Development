# Batch Prediction System Solution

## System Architecture

```
[Input Data] → [Data Validation] → [Feature Engineering] → [Model Prediction] → [Output Storage]
     ↑                 ↑                    ↑                     ↑                   ↑
     |                 |                    |                     |                   |
[Data Lake] ← [Data Quality] ← [Feature Store] ← [Model Registry] ← [Results DB]
     ↑                 ↑                    ↑                     ↑                   ↑
[Raw Data] → [Data Pipeline] → [Training Pipeline] → [Model Versioning] → [Reports]
```

## Implementation

### 1. Batch Processor (processors/batch_processor.py)

```python
from typing import List, Dict
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import mlflow

class BatchProcessor:
    def __init__(self, spark_session: SparkSession):
        self.spark = spark_session
        self.mlflow_client = mlflow.tracking.MlflowClient()
    
    def process_batch(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
        model_version: str
    ) -> Dict[str, float]:
        # Load data
        df = self.spark.read.parquet(input_path)
        
        # Validate data
        validated_df = self._validate_data(df)
        
        # Feature engineering
        features_df = self._engineer_features(validated_df)
        
        # Load model
        model = self._load_model(model_name, model_version)
        
        # Make predictions
        predictions = self._make_predictions(features_df, model)
        
        # Save results
        self._save_results(predictions, output_path)
        
        # Generate metrics
        metrics = self._calculate_metrics(predictions)
        
        return metrics
    
    def _validate_data(self, df):
        # Implement data validation logic
        return df.dropna()
    
    def _engineer_features(self, df):
        # Implement feature engineering logic
        return df
    
    def _load_model(self, model_name: str, version: str):
        return mlflow.pyfunc.load_model(
            f"models:/{model_name}/{version}"
        )
    
    def _make_predictions(self, df, model):
        # Convert to pandas for prediction
        pandas_df = df.toPandas()
        predictions = model.predict(pandas_df)
        
        # Convert back to Spark DataFrame
        predictions_df = self.spark.createDataFrame(
            pd.DataFrame(predictions, columns=["prediction"])
        )
        
        return df.join(predictions_df)
    
    def _save_results(self, predictions_df, output_path: str):
        predictions_df.write.parquet(output_path)
    
    def _calculate_metrics(self, predictions_df) -> Dict[str, float]:
        # Calculate batch processing metrics
        return {
            "total_records": predictions_df.count(),
            "processing_time": 0.0  # Implement actual timing
        }
```

### 2. Airflow DAG (dags/batch_prediction_dag.py)

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from processors.batch_processor import BatchProcessor
from pyspark.sql import SparkSession

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def process_batch(**context):
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("BatchPrediction") \
        .getOrCreate()
    
    # Initialize processor
    processor = BatchProcessor(spark)
    
    # Get execution date
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')
    
    # Process batch
    metrics = processor.process_batch(
        input_path=f"/data/input/{date_str}",
        output_path=f"/data/output/{date_str}",
        model_name="production_model",
        model_version="1"
    )
    
    # Push metrics
    context['task_instance'].xcom_push(
        key='batch_metrics',
        value=metrics
    )

with DAG(
    'batch_prediction',
    default_args=default_args,
    description='Batch prediction pipeline',
    schedule_interval='@daily'
) as dag:
    
    process_task = PythonOperator(
        task_id='process_batch',
        python_callable=process_batch,
        provide_context=True
    )
```

### 3. Model Registry (models/registry.py)

```python
from typing import Dict, Optional
import mlflow
from datetime import datetime

class ModelRegistry:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Dict[str, float]
    ) -> str:
        # Register model
        result = mlflow.register_model(
            model_path,
            model_name
        )
        
        # Log metrics
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
        
        return result.version
    
    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None
    ) -> str:
        if stage:
            versions = self.client.get_latest_versions(
                model_name,
                stages=[stage]
            )
        else:
            versions = self.client.get_latest_versions(model_name)
        
        return versions[0].version if versions else None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
```

### 4. Configuration (config/batch_config.yml)

```yaml
spark:
  app_name: "BatchPrediction"
  master: "yarn"
  executor_memory: "4g"
  executor_cores: 2
  num_executors: 10

mlflow:
  tracking_uri: "http://mlflow:5000"
  registry_uri: "postgresql://mlflow@localhost/mlflow"

data:
  input_base_path: "/data/input"
  output_base_path: "/data/output"
  schema:
    - name: "feature1"
      type: "float"
    - name: "feature2"
      type: "float"
    - name: "feature3"
      type: "string"

model:
  name: "production_model"
  stage: "Production"
  metrics_threshold:
    accuracy: 0.8
    latency_ms: 100
```

## Performance Optimization

1. **Data Processing**
   - Partition data by date
   - Use appropriate file formats (Parquet)
   - Optimize Spark configuration

2. **Resource Management**
   - Dynamic resource allocation
   - Memory tuning
   - Executor configuration

3. **Monitoring**
   - Batch processing time
   - Resource utilization
   - Error rates
   - Data quality metrics

## Testing

### Unit Tests (tests/test_batch_processor.py)

```python
import pytest
from pyspark.sql import SparkSession
from processors.batch_processor import BatchProcessor

@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \
        .appName("TestBatchProcessor") \
        .master("local[*]") \
        .getOrCreate()

@pytest.fixture
def processor(spark):
    return BatchProcessor(spark)

def test_data_validation(processor, spark):
    # Create test data
    data = [(1.0, None), (2.0, 3.0)]
    df = spark.createDataFrame(data, ["col1", "col2"])
    
    # Validate
    validated_df = processor._validate_data(df)
    
    assert validated_df.count() == 1
    assert validated_df.collect()[0][1] == 3.0

def test_feature_engineering(processor, spark):
    # Test feature engineering logic
    pass

def test_end_to_end_processing(processor):
    metrics = processor.process_batch(
        "test_data/input",
        "test_data/output",
        "test_model",
        "1"
    )
    
    assert "total_records" in metrics
    assert metrics["total_records"] > 0
```

### Integration Tests (tests/test_integration.py)

```python
import pytest
from airflow.models import DagBag
from processors.batch_processor import BatchProcessor
from models.registry import ModelRegistry

def test_dag_loading():
    dag_bag = DagBag()
    dag = dag_bag.get_dag(dag_id='batch_prediction')
    assert dag is not None
    assert len(dag.tasks) > 0

def test_end_to_end_pipeline():
    # Test entire pipeline
    pass
```

## Error Handling and Recovery

```python
class BatchProcessingError(Exception):
    pass

class DataValidationError(BatchProcessingError):
    pass

class ModelLoadError(BatchProcessingError):
    pass

def process_with_retry(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return func(*args, **kwargs)
            except BatchProcessingError as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise e
                time.sleep(60)  # Wait before retry
    
    return wrapper

@process_with_retry
def process_batch(input_path: str, output_path: str):
    # Implementation
    pass
```

## Monitoring and Alerting

### Prometheus Configuration (monitoring/prometheus.yml)

```yaml
global:
  scrape_interval: 1m

scrape_configs:
  - job_name: 'batch_metrics'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'spark_metrics'
    static_configs:
      - targets: ['localhost:8080']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

rules:
  - alert: BatchProcessingDelay
    expr: batch_processing_duration_seconds > 3600
    for: 5m
    labels:
      severity: warning
    annotations:
      description: "Batch processing taking longer than expected"
```

### Grafana Dashboard (monitoring/grafana/dashboard.json)

```json
{
  "dashboard": {
    "panels": [
      {
        "title": "Batch Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "batch_processing_duration_seconds"
          }
        ]
      },
      {
        "title": "Records Processed",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(records_processed_total)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(processing_errors_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Documentation

### API Documentation

```python
from pydantic import BaseModel
from typing import List, Dict

class BatchPredictionRequest(BaseModel):
    input_path: str
    output_path: str
    model_name: str
    model_version: str
    config: Dict[str, any]

class BatchPredictionResponse(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, float]
    errors: List[str]

class BatchPredictionAPI:
    """
    API for managing batch predictions.
    
    Endpoints:
    - POST /batch/predict: Submit a batch prediction job
    - GET /batch/{job_id}: Get job status and results
    - GET /batch/metrics: Get aggregated metrics
    """
    pass
```

### Deployment Guide

1. Prerequisites
   - Apache Spark cluster
   - MLflow server
   - Apache Airflow
   - Prometheus & Grafana

2. Configuration
   - Set up environment variables
   - Configure Spark cluster
   - Set up MLflow tracking server

3. Deployment Steps
   - Deploy DAGs
   - Register models
   - Configure monitoring

4. Maintenance
   - Log rotation
   - Backup strategy
   - Scaling guidelines 