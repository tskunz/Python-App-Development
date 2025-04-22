# Production Deployment

## 1. Containerization with Docker

### Dockerfile
```dockerfile
# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/app
    depends_on:
      - db
    volumes:
      - .:/app
    
  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=app
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

## 2. CI/CD Pipeline

### GitHub Actions
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest --cov=app tests/
    
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Deploy to production
      run: |
        # Add deployment steps here
```

## 3. Infrastructure as Code

### Terraform Configuration
```hcl
provider "aws" {
  region = "us-west-2"
}

resource "aws_ecs_cluster" "main" {
  name = "app-cluster"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "app"
  container_definitions    = jsonencode([
    {
      name  = "app"
      image = "app:latest"
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
    }
  ])
}

resource "aws_ecs_service" "app" {
  name            = "app"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
}
```

## 4. Monitoring and Logging

### Prometheus Configuration
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'app'
    static_configs:
      - targets: ['localhost:8000']
```

### Application Metrics
```python
from prometheus_client import Counter, Histogram
from functools import wraps
import time

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total app requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

def monitor(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        start_time = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    return wrapper

@app.get("/users/")
@monitor
async def list_users():
    # Implementation
    pass
```

### Logging Configuration
```python
import logging
import structlog
from pythonjsonlogger import jsonlogger

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter()
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()
```

## 5. Scaling and High Availability

### Load Balancer Configuration
```python
from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)

# AWS Lambda function
def lambda_handler(event, context):
    return handler(event, context)
```

### Cache Configuration
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
```

## 6. Security Best Practices

### SSL Configuration
```python
import ssl
import uvicorn

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('cert.pem', keyfile='key.pem')

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=443,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )
```

### Security Headers
```python
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

app = FastAPI()

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
```

## Interview Questions to Ask

### Infrastructure Questions
1. "What cloud providers do you use?"
2. "How do you handle database migrations in production?"
3. "What's your strategy for scaling the application?"
4. "How do you handle secrets management?"
5. "What's your disaster recovery plan?"

### Deployment Questions
1. "What's your deployment strategy (blue-green, canary, etc.)?"
2. "How do you handle rollbacks?"
3. "What's your approach to zero-downtime deployments?"
4. "How do you manage configuration across environments?"
5. "What monitoring tools do you use?"

### Security Questions
1. "How do you handle security updates?"
2. "What's your approach to vulnerability scanning?"
3. "How do you manage access control?"
4. "What's your incident response process?"
5. "How do you handle data backups?"

## Resources
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform Documentation](https://www.terraform.io/docs)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [OWASP Security Practices](https://owasp.org/www-project-web-security-testing-guide/) 