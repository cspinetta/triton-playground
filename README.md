# Triton Playground

A hands-on playground to explore **Triton Inference Server**, **model serving**, and **ML infrastructure fundamentals** using `ResNet50`, Docker Compose, Prometheus, and Grafana.

## What This Is

This project simulates a real-world ML inference service:

- Serves an ONNX ResNet50 model using **NVIDIA Triton Inference Server**
- Accepts image input and returns the **top predicted class**
- Includes **real-time monitoring** with Prometheus and Grafana
- Runs locally via Docker Compose — no cloud required

## Stack

| Component      | Purpose                            |
|----------------|-------------------------------------|
| Triton Server  | ML model serving engine             |
| ResNet50 (ONNX)| Image classification model          |
| Python Client  | Sends inference requests            |
| Prometheus     | Scrapes and stores Triton metrics   |
| Grafana        | Visualizes request/latency metrics  |
| Docker Compose | Simplified multi-service setup      |

## How to Run It Locally

### 1. Clone the Repo

    git clone https://github.com/cspinetta/triton-playground.git
    cd triton-playground

### 2. Install Python Dependencies (for the client)

    pip install -r requirements.txt

### 3. Download a Sample Image

    curl -L -o sample.jpg https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/RoyalNefertt_Serket_of_AchetAton.jpg/2560px-RoyalNefertt_Serket_of_AchetAton.jpg

### 4. Start All Services

    docker-compose up

Triton will launch and load the ResNet50 model automatically.

## Run a Test Inference

Once the server is running:

    python client_infer.py

Expected output:

    Predicted class: Egyptian_cat (ID: 285)

## Monitoring with Grafana

### 1. Open Grafana in your browser

    http://localhost:3000  
    (Username: admin | Password: admin)

### 2. Add Prometheus Data Source

- Go to ⚙️ Settings → Data Sources
- Click “Add data source” → Prometheus
- Set URL: `http://prometheus:9090`
- Save & Test

### 3. Import the Dashboard

- Click the “+” icon → Import
- Upload `monitoring/triton-dashboard.json`
- Choose Prometheus as the data source
- Click **Import**

You'll see:

- Total inference requests
- Inference success count
- Average latency
- GPU utilization (if applicable)

## 🔒 Requirements

- Docker + Docker Compose
- Python 3.8+
- No GPU required (CPU mode supported)

## 📜 License

MIT

## 🙌 Acknowledgments

- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
