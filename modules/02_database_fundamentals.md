# Database Fundamentals for Python Applications

## SQL Databases

### 1. Core Concepts
- ACID Properties
- Normalization
- Relationships (One-to-One, One-to-Many, Many-to-Many)
- Indexes and Performance
- Transactions

### 2. SQL Operations
- Basic CRUD Operations
- JOINs (INNER, LEFT, RIGHT, FULL)
- Aggregations (GROUP BY, HAVING)
- Subqueries
- Views

### 3. PostgreSQL with Python
```python
import psycopg2
from typing import List, Dict, Any

class DatabaseManager:
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.conn = None
        self.cur = None

    def __enter__(self):
        self.conn = psycopg2.connect(**self.connection_params)
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cur:
            self.cur.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        self.cur.execute(query, params)
        return self.cur.fetchall()

# Usage Example
db_params = {
    'dbname': 'your_db',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost'
}

def get_user_orders(user_id: int) -> List[Dict[str, Any]]:
    query = """
    SELECT o.order_id, o.order_date, p.product_name, oi.quantity
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.user_id = %s
    """
    
    with DatabaseManager(db_params) as db:
        results = db.execute_query(query, (user_id,))
        
    return [
        {
            'order_id': r[0],
            'order_date': r[1],
            'product_name': r[2],
            'quantity': r[3]
        }
        for r in results
    ]
```

## NoSQL Databases

### 1. Core Concepts
- Document-based Storage
- Schema-less Design
- Scaling (Horizontal vs. Vertical)
- CAP Theorem
- Eventual Consistency

### 2. MongoDB with Python
```python
from pymongo import MongoClient
from typing import List, Dict, Any

class MongoManager:
    def __init__(self, connection_uri: str, database: str):
        self.client = MongoClient(connection_uri)
        self.db = self.client[database]

    def insert_document(self, collection: str, document: Dict[str, Any]) -> str:
        result = self.db[collection].insert_one(document)
        return str(result.inserted_id)

    def find_documents(self, collection: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(self.db[collection].find(query))

    def update_document(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> int:
        result = self.db[collection].update_many(query, {'$set': update})
        return result.modified_count

# Usage Example
def process_user_activity(user_id: str, activity: Dict[str, Any]) -> None:
    mongo = MongoManager('mongodb://localhost:27017', 'user_analytics')
    
    # Insert activity
    activity_doc = {
        'user_id': user_id,
        'timestamp': datetime.utcnow(),
        **activity
    }
    
    mongo.insert_document('user_activities', activity_doc)
    
    # Update user stats
    stats_update = {
        'last_activity': datetime.utcnow(),
        'activity_count': {'$inc': 1}
    }
    
    mongo.update_document('user_stats', {'user_id': user_id}, stats_update)
```

## Database Design Patterns

### 1. SQL Design Patterns
- Repository Pattern
- Unit of Work
- Active Record
- Data Mapper

### 2. NoSQL Design Patterns
- Embedded Documents
- References
- Bucket Pattern
- Extended Reference Pattern

## Best Practices

1. **Security**
   - Use parameterized queries
   - Implement proper authentication
   - Encrypt sensitive data
   - Regular backups

2. **Performance**
   - Proper indexing
   - Query optimization
   - Connection pooling
   - Caching strategies

3. **Data Integrity**
   - Validation before storage
   - Consistent error handling
   - Transaction management
   - Data backup and recovery

## Exercises

1. Design and implement a simple e-commerce database schema
2. Create a data access layer using the Repository pattern
3. Implement a caching system for frequently accessed data
4. Build a simple analytics system using MongoDB

## Resources
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Database Design Patterns](https://www.mongodb.com/blog/post/building-with-patterns-a-summary) 