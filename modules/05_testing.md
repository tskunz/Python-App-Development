# Testing and Quality Assurance

## 1. Types of Testing

### Unit Testing
```python
import pytest
from unittest.mock import Mock, patch
from myapp.user_service import UserService

def test_user_creation():
    # Arrange
    user_repo = Mock()
    user_service = UserService(user_repo)
    user_data = {"username": "test", "email": "test@example.com"}
    
    # Act
    user = user_service.create_user(user_data)
    
    # Assert
    assert user.username == "test"
    user_repo.save.assert_called_once_with(user)

@patch('myapp.services.email_service.send_email')
def test_user_notification(mock_send_email):
    # Test with mocked dependencies
    service = UserService()
    service.notify_user("test@example.com", "Hello")
    mock_send_email.assert_called_once()
```

### Integration Testing
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

@pytest.fixture
def db_session():
    # Setup test database
    engine = create_engine('sqlite:///test.db')
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    
    session = SessionLocal()
    yield session
    
    # Cleanup
    session.close()
    Base.metadata.drop_all(engine)

def test_user_workflow(db_session):
    # Test complete user workflow
    user = create_test_user(db_session)
    response = client.get(f"/users/{user.id}")
    assert response.status_code == 200
    
    # Test related operations
    response = client.post(f"/users/{user.id}/orders")
    assert response.status_code == 201
```

### Performance Testing
```python
import time
import statistics
from locust import HttpUser, task, between

class UserBehavior(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def list_users(self):
        start_time = time.time()
        response = self.client.get("/users/")
        duration = time.time() - start_time
        
        assert response.status_code == 200
        assert duration < 0.5  # Response time under 500ms
```

## 2. Test Coverage

### Coverage Configuration
```ini
# .coveragerc
[run]
source = myapp
omit = tests/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
```

### Running Coverage
```python
# pytest.ini
[pytest]
addopts = --cov=myapp --cov-report=html --cov-fail-under=90
```

## 3. Test Fixtures and Factories

### Pytest Fixtures
```python
import pytest
from typing import Dict, Any

@pytest.fixture
def user_data() -> Dict[str, Any]:
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "securepass123"
    }

@pytest.fixture
def authenticated_client(client, user_data):
    # Create and login user
    client.post("/auth/register", json=user_data)
    response = client.post("/auth/login", json=user_data)
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return client
```

### Factory Pattern
```python
from factory import Factory, Faker
from myapp.models import User

class UserFactory(Factory):
    class Meta:
        model = User
    
    username = Faker('user_name')
    email = Faker('email')
    full_name = Faker('name')
    
    @classmethod
    def _create(cls, model_class, *args, **kwargs):
        obj = model_class(*args, **kwargs)
        obj.save()
        return obj
```

## 4. Mocking and Patching

### Mock Objects
```python
from unittest.mock import Mock, patch, MagicMock

def test_payment_processing():
    mock_payment_gateway = Mock()
    mock_payment_gateway.process_payment.return_value = {
        "status": "success",
        "transaction_id": "123"
    }
    
    service = PaymentService(payment_gateway=mock_payment_gateway)
    result = service.process_order(100)
    
    assert result["status"] == "success"
    mock_payment_gateway.process_payment.assert_called_once_with(100)
```

### Context Managers
```python
from contextlib import contextmanager
from unittest.mock import patch

@contextmanager
def mock_database():
    with patch('myapp.database.Session') as mock_session:
        yield mock_session

def test_database_operations():
    with mock_database() as db:
        db.query.return_value.filter.return_value.first.return_value = None
        # Test database operations
```

## 5. Test-Driven Development (TDD)

### TDD Cycle
1. Write a failing test
2. Write minimal code to pass
3. Refactor
4. Repeat

```python
# Step 1: Write failing test
def test_calculate_total():
    cart = ShoppingCart()
    cart.add_item({"price": 10, "quantity": 2})
    assert cart.calculate_total() == 20

# Step 2: Write minimal code
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        self.items.append(item)
    
    def calculate_total(self):
        return sum(item["price"] * item["quantity"] for item in self.items)
```

## 6. Quality Assurance Best Practices

### Code Quality Tools
```python
# pylint configuration
# .pylintrc
[MASTER]
ignore=CVS
persistent=yes
load-plugins=

[MESSAGES CONTROL]
disable=C0111,R0903,C0103

[FORMAT]
max-line-length=120
```

### Type Checking
```python
from typing import List, Optional
from mypy import run

class DataProcessor:
    def process_items(self, items: List[dict]) -> Optional[List[dict]]:
        try:
            return [self._process_item(item) for item in items]
        except Exception:
            return None
    
    def _process_item(self, item: dict) -> dict:
        return {**item, "processed": True}
```

## Interview Questions to Ask

### Testing Strategy Questions
1. "What's your approach to test coverage?"
2. "How do you handle integration testing?"
3. "What's your strategy for performance testing?"
4. "How do you handle test data management?"
5. "What's your approach to automated testing?"

### Quality Assurance Questions
1. "What code quality tools do you use?"
2. "How do you handle code reviews?"
3. "What's your CI/CD pipeline like?"
4. "How do you handle technical debt?"
5. "What's your approach to documentation?"

### Process Questions
1. "How do you prioritize what to test?"
2. "What's your bug tracking process?"
3. "How do you handle test maintenance?"
4. "What's your release validation process?"

## Resources
- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing with Pytest](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)
- [Factory Boy Documentation](https://factoryboy.readthedocs.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html) 