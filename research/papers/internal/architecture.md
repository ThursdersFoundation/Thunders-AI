# Architecture Specification (`ARCHITECTURE.md`)

This document provides a detailed overview of the system architecture, core engine design, data flow pipelines, and deployment topology for **Thunders AI**.

## 1. System Overview

Thunders AI is engineered as a decoupled, multi-layered framework designed to handle real-time Large Language Model (LLM) inference, multimodal data fusion, autonomous robotics control (ROS2), and edge-to-cloud scaling with minimal overhead.

```
+-----------------------------------------------------------------------+
|                            Client Layer                               |
|        [ Web UI ]    [ Mobile App ]    [ ROS2 Nodes ]    [ CLI ]      |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                         API Gateway / Ingress                         |
|      [ FastAPI REST ]    [ gRPC Streaming ]    [ WebSocket Sync ]     |
|             └─────── Auth Middleware (JWT / API Key) ───────┘         |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                           Core Orchestrator                           |
|       [ Task Scheduler ]    [ Memory Buffer ]    [ Model Router ]     |
+-----------------------------------------------------------------------+
        │                          │                           │
        ▼                          ▼                           ▼
+---------------+        +-------------------+        +-----------------+
| LLM Engine    |        | Multimodal Core   |        | Robotics Engine |
| - Transformers|        | - Vision (OpenCV) |        | - SLAM Mapping  |
| - vLLM Backend|        | - Audio/Speech    |        | - Path Planning |
| - Quantization|        | - Cross-Attention |        | - Sensor Fusion |
+---------------+        +-------------------+        +-----------------+
        │                          │                           │
        └──────────────────────────┼───────────────────────────┘
                                   ▼
+-----------------------------------------------------------------------+
|                    Hardware Acceleration Layer                        |
|       [ CUDA / TensorRT ]    [ ONNX Runtime ]    [ Edge Hardware ]    |
+-----------------------------------------------------------------------+

```

## 2. Core Modules Architecture

### 2.1 API & Transport Layer

* **REST Services:** Exposes OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`) via FastAPI.
* **gRPC Server:** Facilitates low-latency inter-process communication (IPC) for microservices and robotic control systems.
* **WebSocket Server:** Delivers real-time bidirectional token streaming and telemetry output.

### 2.2 Execution Engine & Orchestrator

* **Dynamic Scheduler:** Implements continuous batching and priority queues to process incoming requests concurrently without memory contention.
* **Memory Pipeline:** Manages long-term key-value storage (Vector DB) and short-term conversational context buffers.
* **Sandbox Runtime:** Isolate model-generated code execution inside secure containers to prevent security exploits.

### 2.3 Subsystem Engines

* **LLM Engine:** Wraps HuggingFace Transformers, TensorRT-LLM, and vLLM backends, optimizing model weights via INT8/FP16 quantization.
* **Multimodal Engine:** Combines vision and audio inputs into unified embeddings using cross-attention fusion layers.
* **Robotics Module:** Integrates directly with ROS2 nodes, taking spatial embeddings to execute path planning, obstacle avoidance, and SLAM mapping.

## 3. Data Flow Pipelines

### 3.1 LLM Request Processing Pipeline

```
[ User Input ] ──► [ Tokenizer ] ──► [ Context Buffer Retrieval ]
                                             │
                                             ▼
[ Streaming Output ] ◄── [ Engine ] ◄── [ Vector Embeddings ]

```

1. **Input Ingestion:** Request is payload-validated and passed through authentication middleware.
2. **Context Enrichment:** The query is tokenized, and relevant long-term memory records are pulled from the vector index.
3. **Model Inference:** Tokens are pushed to the continuous batching scheduler, dispatched to the CUDA runtime, and generated iteratively.
4. **Response Delivery:** Tokens stream directly back via WebSockets or chunked HTTP responses.

## 4. Security & Isolation Architecture

* **Authentication:** Multi-tier authorization using API Keys and OAuth2 JWT tokens with Role-Based Access Control (RBAC).
* **AI Execution Sandbox:** Code generation tasks execute inside ephemeral Docker containers isolated from host networks.
* **Data Encryption:** TLS 1.3 for data-in-transit; AES-256 for persistent vector storage and checkpoint files.

## 5. Scalability & Deployment Topology

```
                  [ Ingress Controller (Nginx / Traefik) ]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
  [ Worker Node 1 (GPU Cluster) ]         [ Worker Node 2 (Edge Node) ]
  ├── Core Engine Subsystem               ├── Lightweight Runtime
  ├── TensorRT Execution                 └── TensorRT-Edge Engine
  └── Local Cache (Redis)

```

* **Kubernetes Orchestrator:** Uses custom CRDs to auto-scale worker pods based on GPU utilization metrics.
* **Edge Deployment:** Supports low-power ARM architectures (e.g., NVIDIA Jetson) using quantized ONNX runtime models.
