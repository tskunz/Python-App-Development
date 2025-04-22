# Application Architecture and Design Patterns

## 1. Architectural Patterns

### Layered Architecture
```python
# Example of layered architecture
class UserRepository:
    def get_user(self, user_id: int) -> Dict[str, Any]:
        pass

class UserService:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    def get_user_details(self, user_id: int) -> Dict[str, Any]:
        user = self.repository.get_user(user_id)
        # Add business logic here
        return user

class UserController:
    def __init__(self, service: UserService):
        self.service = service

    def get_user(self, user_id: int) -> Response:
        try:
            user = self.service.get_user_details(user_id)
            return Response(data=user, status=200)
        except Exception as e:
            return Response(error=str(e), status=500)
```

### Microservices Architecture
- Service Discovery
- API Gateway
- Message Queues
- Load Balancing
- Circuit Breakers

### Event-Driven Architecture
```python
from typing import Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    type: str
    data: dict
    timestamp: datetime = datetime.utcnow()

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def publish(self, event: Event):
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                handler(event)
```

## 2. Design Patterns

### Creational Patterns
- Singleton
- Factory
- Builder
- Dependency Injection

```python
from typing import Type, TypeVar, Dict

T = TypeVar('T')

class Container:
    _instances: Dict[Type, object] = {}

    @classmethod
    def register(cls, interface: Type[T], implementation: Type[T]):
        cls._instances[interface] = implementation()

    @classmethod
    def resolve(cls, interface: Type[T]) -> T:
        return cls._instances[interface]
```

### Structural Patterns
- Adapter
- Decorator
- Facade
- Proxy

### Behavioral Patterns
- Observer
- Strategy
- Command
- Chain of Responsibility

## 3. SOLID Principles

### Single Responsibility
```python
# Bad
class User:
    def save_to_db(self): pass
    def generate_report(self): pass

# Good
class User:
    pass

class UserRepository:
    def save(self, user: User): pass

class UserReportGenerator:
    def generate(self, user: User): pass
```

### Open/Closed
```python
from abc import ABC, abstractmethod

class PaymentProcessor(ABC):
    @abstractmethod
    def process(self, amount: float): pass

class StripeProcessor(PaymentProcessor):
    def process(self, amount: float):
        # Stripe-specific implementation
        pass

class PayPalProcessor(PaymentProcessor):
    def process(self, amount: float):
        # PayPal-specific implementation
        pass
```

### Liskov Substitution, Interface Segregation, Dependency Inversion
- Follow inheritance rules
- Small, focused interfaces
- Depend on abstractions

## 4. Clean Architecture

### Layers
1. Entities (Domain Models)
2. Use Cases (Application Logic)
3. Interface Adapters (Controllers, Presenters)
4. Frameworks and Drivers (External Interfaces)

```python
# Domain Layer
class User:
    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name

# Use Case Layer
class CreateUserUseCase:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def execute(self, user_data: dict) -> User:
        user = User(**user_data)
        return self.user_repository.save(user)

# Interface Adapter Layer
class UserController:
    def __init__(self, create_user_use_case: CreateUserUseCase):
        self.create_user_use_case = create_user_use_case

    def create_user(self, request_data: dict) -> Response:
        try:
            user = self.create_user_use_case.execute(request_data)
            return Response(data=user, status=201)
        except Exception as e:
            return Response(error=str(e), status=400)
```

## 5. Best Practices

### Configuration Management
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False

    class Config:
        env_file = ".env"
```

### Dependency Management
- Use virtual environments
- Pin dependencies
- Use requirements.txt or pyproject.toml

### Logging and Monitoring
```python
import logging
import structlog

logger = structlog.get_logger()

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ]
    )
```

## Interview Questions to Ask

### Technical Questions
1. "What is the current architecture of the system I'll be working with?"
2. "How do you handle service discovery in your microservices architecture?"
3. "What's your strategy for database migrations and schema changes?"
4. "How do you handle API versioning?"
5. "What monitoring and observability tools do you use?"

### Process Questions
1. "How do you handle technical debt?"
2. "What's the deployment process like?"
3. "How do you handle production incidents?"
4. "What's your code review process?"
5. "How do you make architectural decisions?"

### Team Questions
1. "How is the team structured?"
2. "What's the on-call rotation like?"
3. "How do you handle knowledge sharing?"
4. "What's the process for proposing new technologies?"

## Resources
- [Clean Architecture by Robert Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Martin Fowler's Blog](https://martinfowler.com/)
- [Python Design Patterns](https://python-patterns.guide/)
- [12 Factor App](https://12factor.net/) 