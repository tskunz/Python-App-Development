# Practice Scenarios

## Scenario 1: Event Management System

### Requirements

#### Functional Requirements
1. Users can create and manage events
2. Users can register for events
3. Event creators can track attendance
4. Users can search for events by date, location, and category
5. System should send email notifications for event updates

#### Non-Functional Requirements
1. System should handle 1000 concurrent users
2. Page load time should be under 2 seconds
3. Data should be backed up daily
4. 99.9% uptime requirement
5. All sensitive data must be encrypted

#### Technical Constraints
- Use PostgreSQL for user and event data
- Implement REST API using FastAPI
- Use Redis for caching
- Implement JWT authentication
- Deploy using Docker containers

### Expected Deliverables
1. System architecture diagram
2. Database schema
3. API documentation
4. Implementation code
5. Deployment configuration

## Scenario 2: Data Processing Pipeline

### Requirements

#### Functional Requirements
1. Ingest data from multiple CSV sources
2. Transform and clean data
3. Store processed data in MongoDB
4. Generate daily reports
5. Provide API access to processed data

#### Non-Functional Requirements
1. Process 1 million records per hour
2. Maximum latency of 5 minutes
3. Data consistency checks
4. Error logging and monitoring
5. Automated recovery from failures

#### Technical Constraints
- Use Python 3.8+
- Implement parallel processing
- Use MongoDB for storage
- Implement error handling and retries
- Use Apache Airflow for orchestration

### Expected Deliverables
1. Pipeline architecture diagram
2. Data models
3. Processing scripts
4. Monitoring dashboard
5. Documentation

## Scenario 3: Real-time Analytics Dashboard

### Requirements

#### Functional Requirements
1. Track user interactions in real-time
2. Calculate key metrics (DAU, retention, etc.)
3. Generate custom reports
4. Set up alerts for metric thresholds
5. Export data in multiple formats

#### Non-Functional Requirements
1. Real-time updates (< 1 second delay)
2. Handle 10,000 events per second
3. Historical data available for 12 months
4. Cross-platform compatibility
5. GDPR compliance

#### Technical Constraints
- Use MongoDB for event storage
- Implement websockets for real-time updates
- Use Redis for caching
- Implement data aggregation pipeline
- Use Python for backend processing

### Expected Deliverables
1. System architecture diagram
2. Data flow diagrams
3. API documentation
4. Implementation code
5. Performance test results

## How to Approach These Scenarios

1. **Analysis Phase**
   - Identify key requirements
   - List assumptions
   - Define scope
   - Identify potential challenges

2. **Design Phase**
   - Create system architecture
   - Design database schema
   - Define APIs
   - Plan scaling strategy

3. **Implementation Phase**
   - Set up development environment
   - Implement core features
   - Write tests
   - Document code

4. **Review Phase**
   - Verify requirements
   - Test performance
   - Review security
   - Prepare documentation

## Sample Solution Structure

```
project/
├── src/
│   ├── api/
│   │   ├── routes/
│   │   ├── models/
│   │   └── services/
│   ├── core/
│   │   ├── config/
│   │   └── utils/
│   ├── database/
│   │   ├── migrations/
│   │   └── repositories/
│   └── tests/
├── docs/
│   ├── architecture/
│   └── api/
├── deployment/
│   ├── docker/
│   └── kubernetes/
└── requirements.txt
```

## Evaluation Criteria

1. **Code Quality**
   - Clean, readable code
   - Proper error handling
   - Good documentation
   - Consistent style

2. **Architecture**
   - Scalable design
   - Proper separation of concerns
   - Efficient data flow
   - Security considerations

3. **Performance**
   - Meets performance requirements
   - Efficient resource usage
   - Proper caching
   - Optimization techniques

4. **Documentation**
   - Clear architecture diagrams
   - Detailed API documentation
   - Setup instructions
   - Deployment guide 