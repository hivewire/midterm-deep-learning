# 🧠 Deep Learning Midterm Project  
**Author:** Fuji Aqbal Fadhlillah – 1103223151

---

## 📌 Overview

This repository contains the midterm project for the **Deep Learning** course.  
The project demonstrates the implementation of machine learning and deep learning techniques across **three problem domains**:

1. **Binary Classification** – Fraud Transaction Detection  
2. **Regression** – Continuous Value Prediction  
3. **Clustering** – Customer Segmentation Using Deep Embeddings  

Each task is implemented in its own Jupyter Notebook along with a full preprocessing pipeline, model design, and evaluation workflow.

---

## 📂 Repository Structure

| Notebook | Task | Dataset Domain |
|----------|------|----------------|
| `midterm-DL-1.ipynb` | Binary Classification | Fraud Transaction Detection |
| `midterm-DL-2 regresi.ipynb` | Regression | Numerical Prediction |
| `midterm_DL_3_clustering.ipynb` | Clustering | Customer Segmentation |

---

# 📝 Project Details

---

## 1. 🔍 Fraud Transaction Detection  
**Notebook:** `midterm-DL-1.ipynb`  
**Goal:** Build and evaluate models for detecting fraudulent financial transactions.

### 🔧 Preprocessing Pipeline
- Log transform & engineered temporal/card features  
- Handling class imbalance with **SMOTE**  
- Label encoding for categorical variables  
- Standard scaling

### 🤖 Models Implemented
#### **Deep Learning (PyTorch MLP)**
- Architecture: **256 → 128 → 64**
- ReLU activation  
- BatchNorm + Dropout  

#### **Traditional ML (LightGBM)**
- 64 leaves  
- Max depth = 8  
- `is_unbalance=True`

### 🏋️ Training Techniques
- Optimizer: **AdamW**  
- Scheduler: **CosineAnnealingLR**  
- Early stopping on validation AUC  
- Automatic best-model checkpointing

### 📊 Evaluation
- **ROC-AUC Score** (primary)  
- Validation curve tracking  

---

## 2. 📈 Regression Model  
**Notebook:** `midterm-DL-2 regresi.ipynb`  
**Goal:** Predict a continuous numerical target using a deep learning regression model.

### 🔧 Data Pipeline
- Duplicate removal  
- High-correlation feature filtering (threshold 0.95)  
- Optional PCA & polynomial features  
- Custom train/val/test split (64/16/20)  
- Target scaling via `StandardScaler`

### 🤖 Model Architecture (MLP)
- Hidden layers: **512 → 256 → 128 → 64**  
- ReLU activation  
- BatchNorm + 5% Dropout  
- Optimizer: **AdamW (lr=3e-4)**  
- Scheduler: **CosineAnnealingWarmRestarts**

### 📊 Evaluation Metrics
- MAE  
- RMSE  
- R² Score

**Result:** Achieves approximately **R² ≈ 0.236** on the test dataset.

---

## 3. 🧩 Customer Clustering  
**Notebook:** `midterm_DL_3_clustering.ipynb`  
**Goal:** Cluster customers using unsupervised learning enhanced with deep neural networks.

### 🔧 Preprocessing
- Remove ID column (`CUST_ID`)  
- Median imputation  
- Standard scaling  

### 🤖 Models Implemented
#### **Autoencoder**
- Encoder: **64 → 32 → 10**  
- Decoder: symmetric structure  
- Latent vector used for clustering

#### **K-Means**
- Applied to latent representations

#### **DEC — Deep Embedded Clustering**
- Soft assignment with Student’s t-distribution  
- Joint optimization of reconstruction + clustering loss

### 📉 Evaluation Summary (Clustering)

- **Optimal Clusters:**  
  Using the Elbow Method and Silhouette Score, **k = 4** was selected as the optimal number of clusters.

- **Customer Segments Identified:**  
  The analysis revealed four groups:
  - **Cluster 0:** Moderate spenders with high installment usage and strong payment behavior.  
  - **Cluster 1:** Low spenders who occasionally use cash advances.  
  - **Cluster 2:** High-debt customers with heavy cash-advance usage and low payment-to-minimum ratios.  
  - **Cluster 3:** High spenders with large one-off purchases and high credit limits.

- **Visualization:**  
  PCA was used to reduce features to two components, and a scatter plot showed clear separation between the four clusters.

---

## 🚀 How to Run the Notebooks

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
