#!/usr/bin/env python3
"""
Script to export UML diagrams as PNG images for Jira dashboard.
"""

import os
import subprocess
import json
from pathlib import Path

# Diagram definitions
DIAGRAMS = {
    "system_overview": {
        "title": "System Overview",
        "mermaid": """
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
        """
    },
    
    "sequence_diagram": {
        "title": "Service Interaction Flow",
        "mermaid": """
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
        """
    },
    
    "camera_service": {
        "title": "Camera Service Architecture",
        "mermaid": """
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
        """
    },
    
    "deployment": {
        "title": "Deployment Architecture",
        "mermaid": """
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
        """
    }
}

def create_html_for_diagram(name, title, mermaid_code):
    """Create individual HTML file for each diagram."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 20px; font-family: Arial, sans-serif; }}
        .diagram-container {{ 
            border: 1px solid #ddd; 
            padding: 20px; 
            border-radius: 8px; 
            background: white;
        }}
        h1 {{ color: #2c3e50; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="diagram-container">
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {{ useMaxWidth: true, htmlLabels: true }},
            sequence: {{ useMaxWidth: true }},
            class: {{ useMaxWidth: true }}
        }});
    </script>
</body>
</html>
    """
    return html_content

def export_diagrams():
    """Export all diagrams as HTML files and provide instructions for PNG export."""
    
    # Create exports directory
    exports_dir = Path("docs/exports")
    exports_dir.mkdir(exist_ok=True)
    
    print("🚀 Exporting UML diagrams for Jira...")
    
    # Create individual HTML files
    for name, diagram in DIAGRAMS.items():
        html_content = create_html_for_diagram(name, diagram["title"], diagram["mermaid"])
        
        html_file = exports_dir / f"{name}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created {html_file}")
    
    # Create instructions file
    instructions = """
# How to Export UML Diagrams for Jira

## Option 1: Browser Screenshot (Recommended)
1. Open each HTML file in your browser:
   - docs/exports/system_overview.html
   - docs/exports/sequence_diagram.html
   - docs/exports/camera_service.html
   - docs/exports/deployment.html

2. Take screenshots of each diagram
3. Upload to Jira as images

## Option 2: Browser Developer Tools
1. Open HTML file in browser
2. Right-click on diagram → "Inspect Element"
3. Find the SVG element
4. Right-click → "Save image as..."

## Option 3: Using Puppeteer (Automated)
Run: npm install puppeteer
Then use the provided script to auto-export PNG files.

## Option 4: Mermaid Live Editor
1. Go to https://mermaid.live/
2. Copy diagram code from docs/architecture_uml.md
3. Export as PNG/SVG

## Files Created:
- system_overview.html - High-level architecture
- sequence_diagram.html - Service interactions
- camera_service.html - Camera service details
- deployment.html - Infrastructure setup

## For Jira:
- Upload as images to your Jira dashboard
- Use in documentation pages
- Add to project wikis
- Include in sprint planning
"""
    
    with open(exports_dir / "JIRA_EXPORT_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions)
    
    print(f"✅ Created instructions: {exports_dir}/JIRA_EXPORT_INSTRUCTIONS.md")
    print("\n📋 Next Steps:")
    print("1. Open the HTML files in your browser")
    print("2. Take screenshots of each diagram")
    print("3. Upload the PNG images to your Jira dashboard")
    print("4. Use the diagrams in your project documentation")

if __name__ == "__main__":
    export_diagrams() 