# API Development with Python

## 1. REST API Principles

### Core Concepts
- Resource-based URLs
- HTTP Methods (GET, POST, PUT, DELETE)
- Stateless Communication
- HATEOAS (Hypermedia as the Engine of Application State)
- Status Codes

### Best Practices
1. Use Nouns for Resources
2. Version Your APIs
3. Use HTTP Methods Correctly
4. Implement Proper Error Handling
5. Use Pagination for Large Datasets

## 2. FastAPI Implementation

### Basic Setup
```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Sample API",
    description="API Documentation",
    version="1.0.0"
)

class UserCreate(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class User(UserCreate):
    id: int
    is_active: bool = True

    class Config:
        orm_mode = True
```

### Route Handlers
```python
@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    try:
        # Create user logic here
        return user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    # Get user logic here
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### Dependency Injection
```python
from fastapi import Depends
from typing import Annotated

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/", response_model=List[User])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    users = db.query(UserModel).offset(skip).limit(limit).all()
    return users
```

## 3. Authentication & Authorization

### JWT Implementation
```python
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
```

### Role-Based Access Control
```python
from enum import Enum
from functools import wraps

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

def require_role(role: UserRole):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if user.role != role:
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

## 4. API Documentation

### OpenAPI/Swagger
```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="API Documentation",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "users",
            "description": "Operations with users"
        }
    ]
)

@app.get("/users/", tags=["users"])
async def list_users():
    """
    Retrieve a list of users.
    
    Returns:
        List[User]: List of user objects
    """
    pass
```

## 5. Testing APIs

### Unit Tests
```python
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

def test_create_user():
    response = client.post(
        "/users/",
        json={"username": "test", "email": "test@example.com"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "test"
```

### Integration Tests
```python
@pytest.mark.asyncio
async def test_user_workflow():
    # Create user
    user_data = {"username": "test", "email": "test@example.com"}
    response = await client.post("/users/", json=user_data)
    user_id = response.json()["id"]
    
    # Get user
    response = await client.get(f"/users/{user_id}")
    assert response.status_code == 200
    
    # Update user
    response = await client.put(
        f"/users/{user_id}",
        json={"username": "updated"}
    )
    assert response.status_code == 200
```

## 6. Performance Optimization

### Caching
```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache

@app.get("/expensive-operation")
@cache(expire=60)
async def expensive_operation():
    # Expensive computation here
    return result
```

### Background Tasks
```python
from fastapi import BackgroundTasks

def process_notification(user_id: int, message: str):
    # Send notification logic here
    pass

@app.post("/users/{user_id}/notify")
async def notify_user(
    user_id: int,
    message: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(process_notification, user_id, message)
    return {"status": "Notification scheduled"}
```

## Interview Questions to Ask

### API Design Questions
1. "How do you handle API versioning?"
2. "What's your strategy for API documentation?"
3. "How do you handle breaking changes in APIs?"
4. "What's your approach to API security?"
5. "How do you monitor API performance?"

### Technical Implementation Questions
1. "What's your preferred API framework and why?"
2. "How do you handle rate limiting?"
3. "What's your strategy for API testing?"
4. "How do you handle API authentication?"
5. "What's your approach to API error handling?"

### Process Questions
1. "How do you gather API requirements?"
2. "What's your API review process?"
3. "How do you handle API deprecation?"
4. "What's your strategy for API documentation?"

## Resources
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [REST API Design Best Practices](https://restfulapi.net/)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)
- [AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html) 