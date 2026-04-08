# AI-Fed-FR: AI-Enabled Federated Learning for Fingerprint Recognition

## 📌 Overview

AI-Fed-FR is a novel deep learning framework for fingerprint recognition that integrates:

- Federated Learning (FL) for privacy-preserving training
- PDUSwin-Net (Hybrid Swin Transformer + CNN)
- Sparse Representation-based Denoising (DCT + K-SVD + OMP)

The framework is designed for secure biometric systems where raw fingerprint data remains local.

---

## 🚀 Key Features

- Federated Learning Framework (FedAvg + Reservoir Sampling)
- PDUSwin-Net Architecture (Transformer + CNN hybrid)
- WSQ fingerprint image support
- Sparse denoising using learned dictionaries
- Full evaluation pipeline (ROC, AUC, EER, fairness)

---

## 🏗️ Architecture

<img width="1215" height="790" alt="main" src="https://github.com/user-attachments/assets/79f4bb7a-32c8-42b6-b3b7-f3c9b60550d9" />

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/AI-Fed-FR.git
cd AI-Fed-FR
pip install -r requirements.txt
```

---

## 📁 Dataset

### Overview

| Property | Details |
|---|---|
| **Format** | WSQ (Wavelet Scalar Quantization) |
| **Total Images** | 11,350 |
| **Unique Subjects** | 119 |
| **Sessions** | 2 |
| **Session 1** | 5,950 images |
| **Session 2** | 5,400 images |

### Directory Structure

```
Fingerprint/
├── Session1/    # 5950 files, 5950 images
└── Session2/    # 5400 files, 5400 images
```

### Sample Files

```
7060_l_2_09.wsq
7016_l_2_07.wsq
7003_l_1_09.wsq
```

### Update Dataset Path

```python
DATA_DIR = Path("/your/dataset/path/Fingerprint")
```

> **Note:** Dataset contains WSQ format fingerprint images. Ensure the WSQ PIL plugin is installed before running (`WSQ PIL plugin registered successfully`).

---

## ⚙️ Hardware Specifications

### Compute Environment

| Component | Specification |
|---|---|
| **Primary GPU** | NVIDIA TITAN V |
| **VRAM (Primary)** | 12,288 MiB (~12.64 GB) |
| **CUDA Version** | 11.8 |
| **Driver Version** | 535.288.01 |
| **Active Device** | CUDA |
| **CUDA Benchmark Mode** | Enabled |

> Training was primarily performed on **NVIDIA TITAN V** (GPU 0) with CUDA 11.8 and cuDNN Benchmark mode enabled for optimized performance.

---

## 🏋️ Training

```bash
python Federated_Learning.py
```

---

---## 🔍 Visualization 

![Enhancement Pipeline](./results/Fingerprint enhancement pipeline.png)

**Pipeline Stages:**
1. 📥 **Input:** Original fingerprint image
2. 🧹 **Denoised:** Sparse representation-based denoising
3. ✨ **Enhanced:** PDUSwin-Net output
4. 🎯 **Minutiae:** Detected ridge endings and bifurcations

---

---## 📊 Results

### ROC Curve

> DET curve comparison showing the trade-off between False Match Rate and False Non-Match Rate across all federated clients.

<img width="2958" height="2361" alt="plot2_det_curves" src="https://github.com/user-attachments/assets/ca58b9fe-37c6-4d4c-bcc4-efdac40f2f30" />

---

### Radar Comparison

> Multi-metric radar chart comparing PDUSwin-Net against baseline models across AUC, EER, accuracy, and fairness dimensions.

<img width="2938" height="2498" alt="plot10_radar_comparison" src="https://github.com/user-attachments/assets/3ba22af2-6db8-4dfe-8252-991cc8e39b08" />

---

### Computational Performance

> Comparison of inference time, memory usage, and FLOPs across models to evaluate computational efficiency.

<img width="4161" height="1763" alt="plot9_computational_performance" src="https://github.com/user-attachments/assets/3b0b0ba8-d7c6-43cf-92b8-f517e90ee3a7" />

---

### Performance Comparison

> Bar chart comparing recognition accuracy and AUC of PDUSwin-Net against CNN, Swin Transformer, and other baselines.

<img width="4763" height="1759" alt="plot3_performance_comparison" src="https://github.com/user-attachments/assets/942f5c93-2874-45b7-9a8b-49e86ebee047" />

---

### Training Convergence

> Federated training loss and accuracy curves over communication rounds, showing stable convergence across all clients.

<img width="4161" height="3012" alt="plot4_training_convergence" src="https://github.com/user-attachments/assets/e557ccb9-652c-44b8-b2bb-f46875131fcc" />

---

### Client Fairness

> Per-client performance distribution illustrating equitable model accuracy across all federated participants.

<img width="4162" height="1761" alt="plot8_client_fairness" src="https://github.com/user-attachments/assets/95a4fb69-c558-4092-bd36-038fca0193e5" />

---

### Finger Performance Heatmap

> Heatmap showing recognition accuracy for each finger type, highlighting performance variation across different biometric inputs.

<img width="2826" height="1762" alt="plot7_finger_performance_heatmap" src="https://github.com/user-attachments/assets/beee74da-a99b-4428-9558-29f5c761dc4b" />

---

### Robustness Analysis

> Model performance under various noise levels and image quality degradations, demonstrating the effectiveness of sparse denoising.

<img width="4163" height="2055" alt="plot6_robustness_analysis" src="https://github.com/user-attachments/assets/ec8cc4d2-187e-4440-9f74-c6aee726b694" />

---

## 🔐 Privacy

- No raw data sharing
- Federated distributed learning
- Optional Differential Privacy

---

## 📄 License

MIT License
