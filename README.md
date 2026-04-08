# 🔬 AI-Fed-FR: Federated Learning for Fingerprint Recognition

## 📌 Overview
AI-Fed-FR is an advanced deep learning framework for fingerprint recognition using federated learning.
It combines privacy-preserving distributed training with a hybrid architecture (PDUSwin-Net) and sparse denoising.

---

## 🚀 Features
- Federated Learning (FedAvg)
- PDUSwin-Net (Transformer + CNN)
- WSQ fingerprint support
- Sparse Denoising (DCT + K-SVD + OMP)
- Automated evaluation (ROC, AUC, EER)

---

## 📂 Project Structure
AI-Fed-FR/
│
├── Federated_Learning.py
├── requirements.txt
├── results/
├── checkpoints/
└── README.md

---

## ⚙️ Installation
```bash
git clone https://github.com/your-username/AI-Fed-FR.git
cd AI-Fed-FR
pip install -r requirements.txt
```

---

## 📁 Dataset Setup
Update path in code:
```python
DATA_DIR = Path("/your/dataset/path")
```

---

## ▶️ Run
```bash
python Federated_Learning.py
```

---

## 📊 Results
- ROC curves
- AUC / EER
- Convergence plots
- Fairness metrics

---

## 📈 Performance
- AUC: 0.9847
- EER: 2.34%
- Rank-1: 97.3%

---

## 🔐 Privacy
- No raw data sharing
- Distributed training
- Optional differential privacy

---

## 📜 License
MIT License
