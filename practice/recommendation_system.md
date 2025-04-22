# Real-time Recommendation System Solution

## System Architecture

```
[User Request] → [API Gateway] → [Feature Service] → [Model Service] → [Response]
                      ↑               ↑                    ↑
                      |               |                    |
                [Redis Cache] ← [Feature Store] ← [Model Registry]
                      ↑               ↑                    ↑
              [User Events] → [Event Stream] → [Training Pipeline]
```

## Implementation

### 1. Model Definition (models/recommendation.py)

```python
from typing import List, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import mlflow

class RecommendationModel:
    def __init__(self, model_path: str):
        self.model = mlflow.sklearn.load_model(model_path)
        self.feature_names = self._load_feature_names()
    
    def predict(self, features: Dict[str, float], k: int = 5) -> List[int]:
        feature_vector = self._prepare_features(features)
        distances, indices = self.model.kneighbors(
            feature_vector.reshape(1, -1), 
            n_neighbors=k
        )
        return indices[0].tolist()
    
    def _prepare_features(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features[name] for name in self.feature_names])
    
    def _load_feature_names(self) -> List[str]:
        return mlflow.sklearn.load_model(
            self.model_path + "/feature_names"
        )
```

### 2. Feature Service (services/feature_service.py)

```python
from typing import Dict, Optional
import redis
import json
from fastapi import HTTPException

class FeatureService:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour cache
    
    async def get_user_features(self, user_id: int) -> Dict[str, float]:
        # Try cache first
        cached = self.redis.get(f"user_features:{user_id}")
        if cached:
            return json.loads(cached)
        
        # Calculate features
        features = await self._calculate_user_features(user_id)
        if not features:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Cache features
        self.redis.setex(
            f"user_features:{user_id}",
            self.ttl,
            json.dumps(features)
        )
        return features
    
    async def _calculate_user_features(
        self, user_id: int
    ) -> Optional[Dict[str, float]]:
        # Implementation for feature calculation
        # This would typically involve database queries and computations
        pass
```

### 3. API Layer (api/main.py)

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import List
import mlflow
from services.feature_service import FeatureService
from models.recommendation import RecommendationModel

app = FastAPI()

# Dependency injection
def get_feature_service():
    return FeatureService(redis_url="redis://localhost:6379")

def get_model():
    return RecommendationModel(model_path="models/latest")

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    k: int = 5,
    feature_service: FeatureService = Depends(get_feature_service),
    model: RecommendationModel = Depends(get_model)
) -> List[int]:
    try:
        features = await feature_service.get_user_features(user_id)
        recommendations = model.predict(features, k)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/events")
async def record_event(
    user_id: int,
    item_id: int,
    event_type: str,
    feature_service: FeatureService = Depends(get_feature_service)
):
    # Invalidate cache
    feature_service.redis.delete(f"user_features:{user_id}")
    # Record event (implementation not shown)
    return {"status": "success"}
```

### 4. Monitoring (monitoring/prometheus.yml)

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'recommendation_api'
    static_configs:
      - targets: ['localhost:8000']

  - job_name: 'model_metrics'
    static_configs:
      - targets: ['localhost:8001']
```

### 5. Deployment (deployment/docker-compose.yml)

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models/latest
    volumes:
      - ./models:/models
    depends_on:
      - redis

  redis:
    image: redis:6.2
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  redis_data:
```

## Performance Optimization

1. **Caching Strategy**
   - Feature caching with TTL
   - Model prediction caching for frequent requests
   - Batch prediction support

2. **Load Balancing**
   - Multiple API instances
   - Redis cluster for caching
   - Model service replication

3. **Monitoring Metrics**
   - Response time percentiles
   - Cache hit rates
   - Model prediction latency
   - Error rates

## Testing

### Unit Tests (tests/test_recommendation.py)

```python
import pytest
from models.recommendation import RecommendationModel

@pytest.fixture
def model():
    return RecommendationModel("test_models/latest")

def test_model_prediction(model):
    features = {"feature1": 0.5, "feature2": 0.3}
    predictions = model.predict(features, k=5)
    assert len(predictions) == 5
    assert all(isinstance(p, int) for p in predictions)

def test_invalid_features(model):
    with pytest.raises(KeyError):
        model.predict({"invalid_feature": 0.5})
```

### Integration Tests (tests/test_api.py)

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_recommendations_endpoint():
    response = client.get("/recommendations/1")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_event_recording():
    response = client.post(
        "/events",
        json={"user_id": 1, "item_id": 2, "event_type": "click"}
    )
    assert response.status_code == 200
```

## A/B Testing Framework

```python
from fastapi import Header
import random

class ABTest:
    def __init__(self, experiment_id: str, variants: List[str]):
        self.experiment_id = experiment_id
        self.variants = variants
    
    def get_variant(self, user_id: int) -> str:
        # Consistent hashing for user assignment
        random.seed(f"{self.experiment_id}:{user_id}")
        return random.choice(self.variants)

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    ab_test: ABTest = Depends(get_ab_test)
):
    variant = ab_test.get_variant(user_id)
    if variant == "control":
        model = get_control_model()
    else:
        model = get_experiment_model()
    
    # Rest of the implementation
```

## Model Update Process

1. Train new model version
2. Validate performance metrics
3. Deploy to staging environment
4. Gradual traffic migration
5. Monitor performance
6. Full deployment or rollback

## Error Handling

```python
from fastapi import HTTPException
from typing import Optional

class ModelServiceError(Exception):
    pass

class FeatureServiceError(Exception):
    pass

async def get_recommendations_safe(
    user_id: int,
    feature_service: FeatureService,
    model: RecommendationModel,
    fallback_recommendations: Optional[List[int]] = None
) -> List[int]:
    try:
        features = await feature_service.get_user_features(user_id)
        recommendations = model.predict(features)
        return recommendations
    except FeatureServiceError:
        # Log error and use fallback
        if fallback_recommendations:
            return fallback_recommendations
        raise HTTPException(status_code=503, detail="Service unavailable")
    except ModelServiceError:
        # Log error and use fallback
        if fallback_recommendations:
            return fallback_recommendations
        raise HTTPException(status_code=503, detail="Service unavailable")
```

## Documentation

### API Documentation

```python
from fastapi import FastAPI
from pydantic import BaseModel

class RecommendationResponse(BaseModel):
    items: List[int]
    model_version: str
    latency_ms: float

@app.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    description="Get personalized recommendations for a user",
    responses={
        200: {"description": "Successful response"},
        404: {"description": "User not found"},
        503: {"description": "Service unavailable"}
    }
)
async def get_recommendations(
    user_id: int,
    k: int = Query(5, ge=1, le=100)
) -> RecommendationResponse:
    # Implementation
```

### Monitoring Dashboard (Grafana)

```json
{
  "dashboard": {
    "panels": [
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
``` 