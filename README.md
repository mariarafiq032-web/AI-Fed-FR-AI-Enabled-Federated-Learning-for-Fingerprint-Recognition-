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



## 🛠️ Installation

```bash
git clone https://github.com/your-username/AI-Fed-FR.git
cd AI-Fed-FR
pip install -r requirements.txt


## 📁 Dataset

Update dataset path:

```python
DATA_DIR = Path("/your/dataset/path")

## 🏋️ Training

```bash
python Federated_Learning.py

## 📊 Results

### ROC Curve 

![ROC](results/plot1_roc_curves.png)
<img width="2958" height="2361" alt="plot2_det_curves" src="https://github.com/user-attachments/assets/ca58b9fe-37c6-4d4c-bcc4-efdac40f2f30" />
### radar_comparison

<img width="2938" height="2498" alt="plot10_radar_comparison" src="https://github.com/user-attachments/assets/3ba22af2-6db8-4dfe-8252-991cc8e39b08" />
### computational_performance

<img width="4161" height="1763" alt="plot9_computational_performance" src="https://github.com/user-attachments/assets/3b0b0ba8-d7c6-43cf-92b8-f517e90ee3a7" />

### Comparison
<img width="4763" height="1759" alt="plot3_performance_comparison" src="https://github.com/user-attachments/assets/942f5c93-2874-45b7-9a8b-49e86ebee047" />


### Training_convergence
<img width="4161" height="3012" alt="plot4_training_convergence" src="https://github.com/user-attachments/assets/e557ccb9-652c-44b8-b2bb-f46875131fcc" />

### Fairness

<img width="4162" height="1761" alt="plot8_client_fairness" src="https://github.com/user-attachments/assets/95a4fb69-c558-4092-bd36-038fca0193e5" />
### Finger_performance_heatmap

<img width="2826" height="1762" alt="plot7_finger_performance_heatmap" src="https://github.com/user-attachments/assets/beee74da-a99b-4428-9558-29f5c761dc4b" />
### Robustness_analysis

<img width="4163" height="2055" alt="plot6_robustness_analysis" src="https://github.com/user-attachments/assets/ec8cc4d2-187e-4440-9f74-c6aee726b694" />

## 🔐 Privacy

- No raw data sharing  
- Federated distributed learning  
- Optional Differential Privacy  

## 📄 License
MIT License
