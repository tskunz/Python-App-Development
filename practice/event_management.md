# Event Management System - Sample Solution

## 1. System Architecture

```plaintext
                                   ┌──────────────┐
                                   │   Frontend   │
                                   │    (React)   │
                                   └──────┬───────┘
                                          │
                                          ▼
┌──────────────┐                  ┌──────────────┐
│    Redis     │◄─────────────────│   FastAPI    │
│   (Cache)    │                  │  Backend     │
└──────────────┘                  └──────┬───────┘
                                         │
                                    ┌────┴────┐
                                    │         │
                              ┌────┴─┐   ┌───┴────┐
                              │Redis │   │Postgres │
                              │Queue │   │Database │
                              └──────┘   └────────┘
```

## 2. Database Schema

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Events table
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    location VARCHAR(255) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    creator_id INTEGER REFERENCES users(id),
    category VARCHAR(50) NOT NULL,
    max_attendees INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Event registrations
CREATE TABLE event_registrations (
    event_id INTEGER REFERENCES events(id),
    user_id INTEGER REFERENCES users(id),
    registration_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'registered',
    PRIMARY KEY (event_id, user_id)
);

-- Create indexes
CREATE INDEX idx_events_start_time ON events(start_time);
CREATE INDEX idx_events_category ON events(category);
CREATE INDEX idx_event_registrations_user ON event_registrations(user_id);
```

## 3. Implementation

### Models (models.py)
```python
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    email: EmailStr
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class EventBase(BaseModel):
    title: str
    description: Optional[str]
    location: str
    start_time: datetime
    end_time: datetime
    category: str
    max_attendees: Optional[int]

class EventCreate(EventBase):
    pass

class Event(EventBase):
    id: int
    creator_id: int
    created_at: datetime
    current_attendees: int

    class Config:
        orm_mode = True
```

### Database (database.py)
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/eventdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Services (services/event_service.py)
```python
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from models import Event, EventCreate
from redis import Redis
from datetime import datetime

class EventService:
    def __init__(self, db: Session, cache: Redis):
        self.db = db
        self.cache = cache

    async def create_event(self, event: EventCreate, creator_id: int) -> Event:
        db_event = Event(**event.dict(), creator_id=creator_id)
        self.db.add(db_event)
        self.db.commit()
        self.db.refresh(db_event)
        return db_event

    async def get_events(
        self,
        skip: int = 0,
        limit: int = 10,
        category: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> List[Event]:
        # Try cache first
        cache_key = f"events:{category}:{start_date}:{skip}:{limit}"
        cached_events = self.cache.get(cache_key)
        if cached_events:
            return cached_events

        # Build query
        query = self.db.query(Event)
        if category:
            query = query.filter(Event.category == category)
        if start_date:
            query = query.filter(Event.start_time >= start_date)

        events = query.offset(skip).limit(limit).all()
        
        # Cache results
        self.cache.setex(cache_key, 300, events)  # Cache for 5 minutes
        return events

    async def register_user(self, event_id: int, user_id: int) -> bool:
        event = self.db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Check capacity
        current_registrations = self.db.query(EventRegistration)\
            .filter(EventRegistration.event_id == event_id)\
            .count()

        if event.max_attendees and current_registrations >= event.max_attendees:
            raise HTTPException(status_code=400, detail="Event is full")

        registration = EventRegistration(event_id=event_id, user_id=user_id)
        self.db.add(registration)
        self.db.commit()
        return True
```

### API Routes (routes/events.py)
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from services.event_service import EventService
from models import Event, EventCreate
from auth import get_current_user
from database import get_db
from cache import get_cache

router = APIRouter()

@router.post("/events/", response_model=Event)
async def create_event(
    event: EventCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
):
    service = EventService(db, cache)
    return await service.create_event(event, current_user.id)

@router.get("/events/", response_model=List[Event])
async def list_events(
    skip: int = 0,
    limit: int = 10,
    category: Optional[str] = None,
    start_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
):
    service = EventService(db, cache)
    return await service.get_events(skip, limit, category, start_date)

@router.post("/events/{event_id}/register")
async def register_for_event(
    event_id: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    cache = Depends(get_cache)
):
    service = EventService(db, cache)
    await service.register_user(event_id, current_user.id)
    return {"message": "Successfully registered for event"}
```

## 4. Testing

### Unit Tests (tests/test_event_service.py)
```python
import pytest
from datetime import datetime, timedelta
from services.event_service import EventService
from models import EventCreate

@pytest.fixture
def event_data():
    return {
        "title": "Test Event",
        "description": "Test Description",
        "location": "Test Location",
        "start_time": datetime.utcnow() + timedelta(days=1),
        "end_time": datetime.utcnow() + timedelta(days=1, hours=2),
        "category": "test",
        "max_attendees": 100
    }

def test_create_event(db_session, redis_mock):
    service = EventService(db_session, redis_mock)
    event = EventCreate(**event_data)
    created_event = await service.create_event(event, creator_id=1)
    
    assert created_event.title == event_data["title"]
    assert created_event.creator_id == 1

def test_register_user_full_event(db_session, redis_mock):
    service = EventService(db_session, redis_mock)
    event = EventCreate(**{**event_data, "max_attendees": 1})
    created_event = await service.create_event(event, creator_id=1)
    
    # First registration should succeed
    assert await service.register_user(created_event.id, user_id=2)
    
    # Second registration should fail
    with pytest.raises(HTTPException) as exc:
        await service.register_user(created_event.id, user_id=3)
    assert exc.value.status_code == 400
```

## 5. Deployment

### Docker Compose (docker-compose.yml)
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/eventdb
      - REDIS_URL=redis://cache:6379/0
      - JWT_SECRET=your-secret-key
    depends_on:
      - db
      - cache
    volumes:
      - .:/app

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=eventdb
    volumes:
      - postgres_data:/var/lib/postgresql/data

  cache:
    image: redis:6
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Monitoring Setup (prometheus.yml)
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'event_api'
    static_configs:
      - targets: ['api:8000']
```

This implementation provides:
- Scalable architecture with caching
- Clean separation of concerns
- Type safety with Pydantic models
- Efficient database queries with indexes
- Comprehensive testing setup
- Production-ready deployment configuration 