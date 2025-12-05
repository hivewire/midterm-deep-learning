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
| `midterm_transaction_data_Fuji.ipynb` | Binary Classification | Fraud Transaction Detection |
| `midterm_regresi_ipynb_Fuji_No2.ipynb` | Regression | Numerical Prediction |
| `clustering_midterm_Fuji_No3.ipynb` | Clustering | Customer Segmentation |

---

# 📝 Project Details

---

## 1. 🔍 Fraud Transaction Detection  
**Notebook:** `midterm_transaction_data_Fuji.ipynb`  
**Goal:** Build and evaluate models for detecting fraudulent financial transactions.

### 🔧 Preprocessing Pipeline
#### **Missing Value Handling**
- Dropped columns with over 70% missing values.
- Imputed numerical columns with median.
- imputed categorical columns with the mode.

#### **Feature Engineering:**
- Extracted temporal features from TransactionDT (hour, day of week, day of year).
- Created a composite cardID.
- Calculated Card_Daily_Transaction_Count (daily transaction frequency per card).
- Created TransactionAmt_to_mean_card1 (transaction amount ratio to mean for card1).
- Created TransactionAmt_per_Card_Daily_Transaction_Count.

#### **Categorical Encoding:**
- Label encoding for categorical features using Polars Categorical dtype with global StringCache for consistent mapping.

#### **Class Imbalance Handling:**
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data.

#### **Feature Scaling**
- Standard Scaling applied to numerical features using StandardScaler for the Deep Learning model.

### 🤖 Models Implemented

#### **Deep Learning (TensorFlow/Keras MLP)**
- Architecture: Input(shape) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)

#### **Traditional ML (LightGBM)**
- Model: LGBMClassifier Hyperparameter Tuning: Performed using RandomizedSearchCV (10 iterations, 3-fold cross-validation, roc_auc scoring).
Key tuned parameters: n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample.


### 🏋️ Training Techniques
#### **Deep Learning Model:**
Optimizer: Adam  
Loss Function: binary_crossentropy  
Metrics: accuracy  
Epochs: 10  
Batch Size: 32    

### 📊 Evaluation
**Metrics:** AUC-ROC Score, Precision, Recall. Visualization: ROC Curves, Deep Learning model's training history (loss and accuracy plots).

#### **Result:** 
##### LightGBM Model Performance: 
AUC-ROC Score: 0.9984  
Precision: 0.9995  
Recall: 0.9863   
            
##### Deep Learning Model Performance (after scaling): 
AUC-ROC Score: 0.9982  
Precision: 0.9749  
Recall: 0.9845  

---

## 2. 📈 Regression Model  
**Notebook:** `midterm_regresi_ipynb_Fuji_No2.ipynb`  
**Goal:** Predict the 'Target' (tahun/year) based on provided features.

### 🤖 Model Architecture (MLP)
- Model: LinearRegression
- Hyperparameter Tuning: GridSearchCV was used to find the best parameters: {'fit_intercept': True, 'positive': False}.

### 📊 Evaluation Metrics
- MAE: 6.7784
- RMSE: 9.5228
- R² Score: 0.2360

---

## 3. 🧩 Customer Clustering  
**Notebook:** `clustering_midterm_Fuji_No3.ipynb`  
**Goal:** Cluster customers using unsupervised learning to identify distinct customer segments based on their spending and payment behaviors.

### 🔧 Preprocessing
- Missing Value Handling: CREDIT_LIMIT and MINIMUM_PAYMENTS were imputed using their respective medians.
- Outlier Treatment: Outliers in relevant numerical features were capped using the Interquartile Range (IQR) method. 
- Feature Engineering: Five new features were created to capture key behavioral aspects:    
MONTHLY_AVG_PURCHASES    
PURCHASES_BY_TYPE
CASH_ADVANCE_TO_PURCHASES_RATIO
LIMIT_USAGE    
PAYMENT_TO_MIN_PAYMENT_RATIO

- ID Removal: The CUST_ID column was dropped.
- Standard Scaling: All numerical features were scaled using StandardScaler to ensure equal contribution to the clustering algorithm.

### 🤖 Models Implemented
#### **K-Means**
- Applied to the scaled and engineered features.

### 📉 Evaluation Summary (Clustering)

- **Optimal Clusters:**  
  Using the Elbow Method and Silhouette Score, **k = 4** was selected as the optimal number of clusters.

- **Customer Segments Identified:**  
  The analysis revealed four groups:
  - **Cluster 0:** Moderate spenders with high installment usage and strong payment behavior.  
  - **Cluster 1:** Low spenders who occasionally use cash advances.  
  - **Cluster 2:** High-debt customers with heavy cash-advance usage and low payment-to-minimum ratios.  
  - **Cluster 3:** High spenders with large one-off purchases and high credit limits.
