# SpokHand SLR Microservices Architecture - UML Documentation

## 1. System Overview (High-Level Architecture)

```mermaid
graph TB
    subgraph "Client Layer"
        UI[React Frontend<br/>Unified UI]
    end
    
    subgraph "API Gateway Layer"
        GW[API Gateway<br/>Load Balancer]
    end
    
    subgraph "Microservices Layer"
        subgraph "Core Services"
            CS[Camera Service<br/>Real-time Processing]
            AS[Annotation Service<br/>Sign-Cut Platform]
            US[Upload Service<br/>Data Management]
        end
        
        subgraph "Support Services"
            SS[Storage Service<br/>File & Data Storage]
            MS[ML Service<br/>Sign Recognition]
            AN[Analytics Service<br/>Data Analysis]
        end
    end
    
    subgraph "Infrastructure Layer"
        DB[(PostgreSQL<br/>Metadata)]
        S3[(S3 Storage<br/>Videos & Files)]
        RED[(Redis<br/>Cache & Sessions)]
        MQ[(Message Queue<br/>Async Processing)]
    end
    
    UI --> GW
    GW --> CS
    GW --> AS
    GW --> US
    GW --> SS
    GW --> MS
    GW --> AN
    
    CS --> SS
    CS --> MS
    AS --> SS
    US --> SS
    AN --> SS
    
    CS --> MQ
    AS --> MQ
    US --> MQ
    
    SS --> DB
    SS --> S3
    CS --> RED
    AS --> RED
```

## 2. Service Interaction Sequence Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant G as API Gateway
    participant C as Camera Service
    participant A as Annotation Service
    participant S as Storage Service
    participant M as ML Service
    participant AN as Analytics Service
    
    Note over U,AN: Data Collection Flow
    U->>F: Start Camera Recording
    F->>G: POST /api/camera/start
    G->>C: Initialize Camera Session
    C->>S: Store Session Metadata
    C->>M: Start Hand Tracking
    M->>F: Stream Hand Landmarks
    F->>U: Display Real-time Feed
    
    U->>F: Stop Recording
    F->>G: POST /api/camera/stop
    G->>C: Stop Recording
    C->>S: Save Video File
    C->>AN: Log Recording Event
    S->>F: Return Video URL
    F->>U: Show Recording Complete
    
    Note over U,AN: Annotation Flow
    U->>F: Upload Video for Annotation
    F->>G: POST /api/upload/video
    G->>S: Store Video File
    S->>A: Queue for Annotation
    A->>F: Return Annotation Interface
    F->>U: Display Annotation Tool
    
    U->>F: Complete Annotation
    F->>G: PUT /api/annotate/{id}
    G->>A: Save Annotations
    A->>S: Store Annotation Data
    A->>AN: Log Annotation Event
    S->>F: Return Success
    F->>U: Show Annotation Complete
```

## 3. Service Class Diagrams

### 3.1 Camera Service
```mermaid
classDiagram
    class CameraService {
        +startSession(cameraType, settings)
        +stopSession(sessionId)
        +getStream(sessionId)
        +recordSign(sessionId, signName)
    }
    
    class CameraSession {
        -sessionId: UUID
        -cameraType: string
        -settings: object
        -status: string
        +initialize()
        +startRecording()
        +stopRecording()
        +getStatus()
    }
    
    class HandTracker {
        -model: MediaPipe
        +trackHands(frame)
        +getLandmarks()
        +calculateConfidence()
    }
    
    class VideoRecorder {
        -outputPath: string
        -codec: string
        +startRecording()
        +stopRecording()
        +saveVideo()
    }
    
    CameraService --> CameraSession
    CameraSession --> HandTracker
    CameraSession --> VideoRecorder
```

### 3.2 Annotation Service
```mermaid
classDiagram
    class AnnotationService {
        +processVideo(videoId)
        +getAnnotations(annotationId)
        +updateAnnotations(annotationId, segments)
        +exportAnnotations(annotationId, format)
    }
    
    class VideoProcessor {
        -videoPath: string
        -metadata: object
        +extractFrames()
        +generateThumbnails()
        +validateFormat()
    }
    
    class AnnotationSegment {
        -segmentId: UUID
        -startTime: timestamp
        -endTime: timestamp
        -segmentType: string
        -description: string
        +validate()
        +toJSON()
    }
    
    class AnnotationSession {
        -sessionId: UUID
        -videoId: UUID
        -segments: AnnotationSegment[]
        -status: string
        +addSegment(segment)
        +removeSegment(segmentId)
        +exportData(format)
    }
    
    AnnotationService --> VideoProcessor
    AnnotationService --> AnnotationSession
    AnnotationSession --> AnnotationSegment
```

### 3.3 Storage Service
```mermaid
classDiagram
    class StorageService {
        +storeVideo(file, metadata)
        +storeAnnotation(data, videoId)
        +getVideo(videoId)
        +getAnnotation(annotationId)
        +deleteVideo(videoId)
    }
    
    class VideoStorage {
        -bucketName: string
        -region: string
        +uploadVideo(file, path)
        +downloadVideo(path)
        +deleteVideo(path)
        +getVideoUrl(path)
    }
    
    class DatabaseManager {
        -connection: Connection
        +insertVideo(metadata)
        +insertAnnotation(data)
        +queryVideos(filters)
        +queryAnnotations(videoId)
    }
    
    class CacheManager {
        -redisClient: Redis
        +setCache(key, value, ttl)
        +getCache(key)
        +deleteCache(key)
        +clearCache()
    }
    
    StorageService --> VideoStorage
    StorageService --> DatabaseManager
    StorageService --> CacheManager
```

## 4. Data Models

### 4.1 Video Entity
```mermaid
erDiagram
    VIDEO {
        uuid id PK
        string filename
        string original_name
        string file_path
        string file_type
        int file_size
        timestamp created_at
        timestamp updated_at
        string status
        json metadata
    }
    
    ANNOTATION {
        uuid id PK
        uuid video_id FK
        string annotation_type
        json segments
        timestamp created_at
        timestamp updated_at
        string status
    }
    
    SESSION {
        uuid id PK
        string session_type
        timestamp start_time
        timestamp end_time
        string status
        json settings
    }
    
    VIDEO ||--o{ ANNOTATION : "has"
    SESSION ||--o{ VIDEO : "creates"
```

### 4.2 API Response Models
```mermaid
classDiagram
    class APIResponse {
        +status: string
        +message: string
        +data: object
        +timestamp: timestamp
    }
    
    class VideoResponse {
        +id: UUID
        +filename: string
        +url: string
        +metadata: object
        +status: string
    }
    
    class AnnotationResponse {
        +id: UUID
        +videoId: UUID
        +segments: array
        +exportUrl: string
        +status: string
    }
    
    class CameraResponse {
        +sessionId: UUID
        +streamUrl: string
        +status: string
        +settings: object
    }
    
    APIResponse --> VideoResponse
    APIResponse --> AnnotationResponse
    APIResponse --> CameraResponse
```

## 5. Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX Load Balancer]
    end
    
    subgraph "API Gateway"
        AG[Kong API Gateway]
    end
    
    subgraph "Microservices Cluster"
        subgraph "Service 1"
            CS1[Camera Service<br/>Instance 1]
            CS2[Camera Service<br/>Instance 2]
        end
        
        subgraph "Service 2"
            AS1[Annotation Service<br/>Instance 1]
            AS2[Annotation Service<br/>Instance 2]
        end
        
        subgraph "Service 3"
            US1[Upload Service<br/>Instance 1]
            US2[Upload Service<br/>Instance 2]
        end
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL<br/>Primary)]
        DB2[(PostgreSQL<br/>Replica)]
        S3[(S3 Storage)]
        RED[(Redis Cluster)]
    end
    
    subgraph "Monitoring"
        MON[Prometheus]
        LOG[ELK Stack]
        TRACE[Jaeger]
    end
    
    LB --> AG
    AG --> CS1
    AG --> CS2
    AG --> AS1
    AG --> AS2
    AG --> US1
    AG --> US2
    
    CS1 --> DB
    CS2 --> DB
    AS1 --> DB
    AS2 --> DB
    US1 --> DB
    US2 --> DB
    
    CS1 --> S3
    CS2 --> S3
    AS1 --> S3
    AS2 --> S3
    US1 --> S3
    US2 --> S3
    
    CS1 --> RED
    CS2 --> RED
    AS1 --> RED
    AS2 --> RED
    US1 --> RED
    US2 --> RED
    
    CS1 --> MON
    CS2 --> MON
    AS1 --> MON
    AS2 --> MON
    US1 --> MON
    US2 --> MON
```

## 6. Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React + TypeScript | Unified UI |
| API Gateway | Kong/NGINX | Load balancing, routing |
| Services | Node.js/Python | Microservices |
| Database | PostgreSQL | Metadata storage |
| Cache | Redis | Session management |
| Storage | AWS S3 | File storage |
| Message Queue | RabbitMQ/Kafka | Async processing |
| Monitoring | Prometheus + Grafana | Metrics |
| Logging | ELK Stack | Log management |
| Tracing | Jaeger | Distributed tracing |

## 7. Security Architecture

```mermaid
graph TB
    subgraph "Security Layer"
        AUTH[Authentication Service]
        AUTHZ[Authorization Service]
        ENC[Encryption Service]
    end
    
    subgraph "API Gateway"
        AG[Kong Gateway]
    end
    
    subgraph "Services"
        S1[Camera Service]
        S2[Annotation Service]
        S3[Upload Service]
    end
    
    AG --> AUTH
    AG --> AUTHZ
    AUTH --> S1
    AUTH --> S2
    AUTH --> S3
    AUTHZ --> S1
    AUTHZ --> S2
    AUTHZ --> S3
    ENC --> S1
    ENC --> S2
    ENC --> S3
```

This architecture provides:
- ✅ Independent scaling of services
- ✅ Fault isolation
- ✅ Technology flexibility
- ✅ Easy maintenance and updates
- ✅ Clear separation of concerns
- ✅ Scalable data processing
- ✅ Real-time capabilities
- ✅ Comprehensive monitoring 