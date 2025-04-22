# Python Fundamentals for Application Development

## Key Concepts

### 1. Data Structures
- Lists, Tuples, Sets, and Dictionaries
- When to use each data structure
- List comprehensions and Dictionary comprehensions
- Advanced operations (map, filter, reduce)

### 2. Object-Oriented Programming
- Classes and Objects
- Inheritance and Polymorphism
- Encapsulation and Abstraction
- Class methods, Static methods, and Properties
- Magic methods

### 3. Error Handling
- Try-except blocks
- Custom exceptions
- Context managers (with statement)
- Debugging techniques

### 4. Functional Programming
- First-class functions
- Lambda functions
- Decorators
- Generators and Iterators

### 5. Modern Python Features
- Type hints and annotations
- Async/await syntax
- f-strings
- Walrus operator (:=)

## Code Examples

### Object-Oriented Programming Example
```python
from typing import List, Optional

class DataProcessor:
    def __init__(self, data: List[dict]):
        self.data = data
        self._processed = False

    @property
    def is_processed(self) -> bool:
        return self._processed

    def process_data(self) -> Optional[List[dict]]:
        try:
            # Example processing
            processed_data = [
                {**item, 'processed': True}
                for item in self.data
            ]
            self._processed = True
            return processed_data
        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    @staticmethod
    def validate_data(data: dict) -> bool:
        required_fields = ['id', 'name', 'value']
        return all(field in data for field in required_fields)
```

### Error Handling Example
```python
class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def process_user_data(data: dict) -> dict:
    try:
        if not isinstance(data, dict):
            raise DataValidationError("Input must be a dictionary")
        
        required_fields = ['username', 'email']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise DataValidationError(f"Missing required fields: {missing_fields}")
            
        # Process the data
        processed_data = {
            'username': data['username'].lower(),
            'email': data['email'].lower(),
            'verified': False
        }
        
        return processed_data
        
    except DataValidationError as e:
        print(f"Validation error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

## Best Practices

1. **Code Organization**
   - Use meaningful variable and function names
   - Follow PEP 8 style guide
   - Keep functions small and focused
   - Use type hints for better code clarity

2. **Performance**
   - Use appropriate data structures
   - Understand time and space complexity
   - Use generators for large datasets
   - Profile code when necessary

3. **Documentation**
   - Write clear docstrings
   - Include examples in documentation
   - Document exceptions and edge cases
   - Use type hints for better IDE support

## Exercises

1. Create a class hierarchy for a data processing pipeline
2. Implement a decorator for timing function execution
3. Write a context manager for database connections
4. Create a generator function for processing large files

## Resources
- [Python Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Design Patterns](https://python-patterns.guide/) 