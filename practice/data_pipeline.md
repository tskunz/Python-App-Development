# Data Processing Pipeline - Sample Solution

## 1. System Architecture

```plaintext
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │    │   Apache    │    │  Processing │    │  MongoDB    │
│   Files     │───►│   Airflow   │───►│  Workers    │───►│  Database   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │                  │
                          │                   │                  │
                          ▼                   ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │   Error     │    │  Monitoring │    │    API      │
                   │   Logs      │    │  Dashboard  │    │  Service    │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

## 2. Implementation

### Data Models (models.py)
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

class DataSource(BaseModel):
    name: str
    file_pattern: str
    schema: Dict[str, str]
    validation_rules: Optional[Dict[str, Any]]

class ProcessedData(BaseModel):
    source_name: str
    processed_at: datetime
    record_count: int
    success_count: int
    error_count: int
    data: List[Dict[str, Any]]

class ProcessingError(BaseModel):
    source_name: str
    error_time: datetime
    error_type: str
    error_message: str
    record: Optional[Dict[str, Any]]
```

### Data Processor (processors/base.py)
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

class DataProcessor(ABC):
    def __init__(self, source: DataSource):
        self.source = source

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def load(self, data: pd.DataFrame) -> bool:
        pass

    def process(self, file_path: str) -> ProcessedData:
        try:
            # Read data
            df = pd.read_csv(file_path)
            
            # Validate
            valid_df = self.validate(df)
            
            # Transform
            transformed_df = self.transform(valid_df)
            
            # Load
            success = self.load(transformed_df)
            
            return ProcessedData(
                source_name=self.source.name,
                processed_at=datetime.utcnow(),
                record_count=len(df),
                success_count=len(transformed_df),
                error_count=len(df) - len(transformed_df),
                data=transformed_df.to_dict('records')
            )
        except Exception as e:
            self._log_error(str(e), file_path)
            raise
```

### CSV Processor (processors/csv_processor.py)
```python
from typing import List, Dict, Any
import pandas as pd
from processors.base import DataProcessor

class CSVProcessor(DataProcessor):
    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply schema validation
        for column, dtype in self.source.schema.items():
            if column not in data.columns:
                raise ValueError(f"Missing required column: {column}")
            data[column] = data[column].astype(dtype)

        # Apply custom validation rules
        if self.source.validation_rules:
            for column, rules in self.source.validation_rules.items():
                if "min" in rules:
                    data = data[data[column] >= rules["min"]]
                if "max" in rules:
                    data = data[data[column] <= rules["max"]]
                if "not_null" in rules and rules["not_null"]:
                    data = data.dropna(subset=[column])

        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add metadata columns
        data['processed_at'] = datetime.utcnow()
        data['source_name'] = self.source.name
        
        # Apply any source-specific transformations
        if 'amount' in data.columns:
            data['amount'] = data['amount'].round(2)
        
        return data

    def load(self, data: pd.DataFrame) -> bool:
        # Convert to MongoDB format and insert
        records = data.to_dict('records')
        result = mongodb_client[self.source.name].insert_many(records)
        return len(result.inserted_ids) == len(records)
```

### Airflow DAG (dags/data_processing_dag.py)
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import glob
import os

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

def process_source_files(source_name: str, **context):
    source = DataSource.get_by_name(source_name)
    processor = CSVProcessor(source)
    
    # Find files matching pattern
    files = glob.glob(source.file_pattern)
    
    for file_path in files:
        try:
            result = processor.process(file_path)
            # Move processed file to archive
            archive_file(file_path)
        except Exception as e:
            # Log error and continue with next file
            log_processing_error(source_name, file_path, str(e))
            continue

dag = DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='Process data from multiple CSV sources',
    schedule_interval=timedelta(hours=1),
    catchup=False
)

# Create tasks for each data source
for source in DataSource.get_all():
    task = PythonOperator(
        task_id=f'process_{source.name}',
        python_callable=process_source_files,
        op_kwargs={'source_name': source.name},
        dag=dag
    )
```

### API Service (api/main.py)
```python
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from datetime import datetime
from models import ProcessedData, ProcessingError

app = FastAPI()

@app.get("/data/{source_name}")
async def get_processed_data(
    source_name: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[ProcessedData]:
    query = {"source_name": source_name}
    if start_date:
        query["processed_at"] = {"$gte": start_date}
    if end_date:
        query["processed_at"]["$lte"] = end_date
    
    data = mongodb_client[source_name].find(query)
    return [ProcessedData(**item) for item in data]

@app.get("/errors/{source_name}")
async def get_processing_errors(
    source_name: str,
    start_date: Optional[datetime] = None
) -> List[ProcessingError]:
    query = {"source_name": source_name}
    if start_date:
        query["error_time"] = {"$gte": start_date}
    
    errors = mongodb_client["errors"].find(query)
    return [ProcessingError(**error) for error in errors]
```

## 3. Monitoring Setup

### Prometheus Configuration (prometheus.yml)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow:8080']

  - job_name: 'processing_api'
    static_configs:
      - targets: ['api:8000']
```

### Grafana Dashboard (dashboard.json)
```json
{
  "dashboard": {
    "title": "Data Processing Pipeline",
    "panels": [
      {
        "title": "Processing Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(processed_records_total[5m])",
            "legendFormat": "{{source_name}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "rate(processing_errors_total[5m])",
            "legendFormat": "{{source_name}}"
          }
        ]
      }
    ]
  }
}
```

## 4. Error Handling and Recovery

### Error Handler (utils/error_handler.py)
```python
import logging
from functools import wraps
from typing import Callable
from models import ProcessingError

def handle_processing_errors(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log error
            error = ProcessingError(
                source_name=kwargs.get('source_name', 'unknown'),
                error_time=datetime.utcnow(),
                error_type=type(e).__name__,
                error_message=str(e),
                record=kwargs.get('record')
            )
            
            # Save to MongoDB
            mongodb_client['errors'].insert_one(error.dict())
            
            # Log to file
            logging.error(f"Processing error: {error.dict()}")
            
            # Retry logic can be implemented here
            raise
    
    return wrapper
```

## 5. Performance Optimization

### Parallel Processing (utils/parallel.py)
```python
from concurrent.futures import ProcessPoolExecutor
from typing import List, Callable
import pandas as pd

def process_in_parallel(
    data: pd.DataFrame,
    process_func: Callable,
    num_workers: int = 4
) -> List:
    # Split dataframe into chunks
    chunk_size = len(data) // num_workers
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_func, chunks))
    
    return results
```

This implementation provides:
- Scalable data processing pipeline
- Error handling and recovery
- Parallel processing capabilities
- Monitoring and alerting
- API access to processed data
- Comprehensive logging 