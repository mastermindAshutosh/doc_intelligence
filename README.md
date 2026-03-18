# Document Intelligence System

A production-grade Document Intelligence system featuring a FastAPI backend with advanced feature fusion, Out-Of-Distribution (OOD) monitoring, and a Streamlit dashboard.

## Features

- **Ingestion & Extraction:** Extract raw content from PDF and DOCX documents with support for metadata analysis.
- **Layout Graph Encoding:** Custom `DocumentGraphEncoder` using Graph Attention Network (GATv2) modules for encoding layout geometry and structure.
- **Dynamic Inference Batching:** Back-end batch server optimizes throughput for classification endpoints.
- **Continuous Monitoring Pipeline:** Real-time dashboards monitor model performance metrics.
- **Confidence Scoring & Calibration:** Reliable scoring based on expected limits with fallback support for uncalibrated pipelines.

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

### Execution

To build and launch both the Backend and Frontend nodes simultaneously:

```bash
docker-compose up --build
```

- **Backend API:** `http://localhost:8000`
- **Dashboard UI:** `http://localhost:8501`

### Testing

Tests can be run locally using pytest in isolation or linked within containerized volumes.

```bash
pytest
```
