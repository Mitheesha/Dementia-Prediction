# 🧠 Dementia Prediction – Machine Learning Project

This repository contains the full machine learning workflow for predicting dementia using structured patient data. The project includes data exploration, feature engineering, preprocessing, model training, hyperparameter tuning, evaluation, and interpretability analysis.

The goal is to build a reliable classification model capable of identifying patients at risk of dementia using non-medical factors.

---

## 📘 Project Overview

Dementia is a progressive neurological condition, and early detection is crucial.  
This project builds a machine learning model to classify patients as **Demented** or **Non-Demented** based on structured input data.

The workflow includes:
- Data exploration and cleaning  
- Feature engineering  
- Handling class imbalance (SMOTE)  
- Model building and optimization  
- Evaluation and comparison  
- Interpretability and insights  

---


## 🔍 Key Features of the Project

### ✔ Exploratory Data Analysis (EDA)
- Summary statistics  
- Missing value analysis  
- Correlation heatmaps  
- Class imbalance visualization  

### ✔ Feature Engineering
- Selection of non-medical features  
- Categorical encoding  
- Feature reduction and cleanup  
- Final feature selection  

### ✔ Data Preprocessing
- Handling missing values  
- Label encoding and scaling  
- Train–test split  
- SMOTE applied **only to training data**  
- Prevention of data leakage  

### ✔ Model Building
Trained and compared models including:
- Logistic Regression  
- Random Forest  
- XGBoost  
- CatBoost  
- Gradient Boosting  

### ✔ Hyperparameter Tuning
Used GridSearchCV / RandomizedSearchCV to optimize model performance.

### ✔ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC-ROC  
- Confusion matrix  

### ✔ Explainability & Interpretability
Without advanced tools like SHAP or LIME, interpretability was achieved via:
- Correlation analysis  
- Feature-type inspection  
- Class distribution insights  
- Missing value interpretation  
- Clinical variable understanding  


---

## 📊 Dataset

The dataset includes demographic, behavioral, and medical indicators.  
The full dataset and data dictionary are located in the `data/` folder in compressed format.

---

## ⚙ Technologies Used

- Python 3.11.7  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- XGBoost  
- CatBoost  
- Jupyter Notebooks  

---

## 🏆 Results

- Class imbalance handled using SMOTE (training set only)  
- Multiple models trained and compared  
- Final model achieved strong recall, F1-score, and ROC-AUC  
- Key insights gained from data distribution and correlations  



---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/<your-username>/Dementia-Prediction.git

# Enter the project directory
cd Dementia-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebooks
jupyter notebook
🤝 Contributing
