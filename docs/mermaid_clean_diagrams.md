# Clean Mermaid Diagrams for Live Editor

## 1. System Overview
Copy this code to https://mermaid.live/

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

## 2. Sequence Diagram
Copy this code to https://mermaid.live/

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

## 3. Camera Service Class Diagram
Copy this code to https://mermaid.live/

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

## 4. Deployment Architecture
Copy this code to https://mermaid.live/

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

## 5. Data Models
Copy this code to https://mermaid.live/

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

## Instructions for Mermaid Live Editor:

1. Go to https://mermaid.live/
2. Clear the editor
3. Copy ONE diagram at a time from above
4. Paste into the editor
5. The diagram should render automatically
6. Use the export button to save as PNG/SVG
7. Repeat for each diagram you need

## Common Issues and Solutions:

- **If you get syntax errors**: Make sure you're copying the code between the ```mermaid and ``` markers
- **If diagram doesn't render**: Try refreshing the page and pasting again
- **If text is too small**: Use the zoom controls in the editor
- **If you need different colors**: Use the theme selector in the editor

## Export Settings for Jira:

- **Format**: PNG (recommended for Jira)
- **Scale**: 1.5x or 2x for better quality
- **Background**: White (works best in Jira)
- **Theme**: Default (clean and professional) 