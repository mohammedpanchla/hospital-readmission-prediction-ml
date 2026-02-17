# hospital-readmission-prediction-ml
Machine learning system to predict 30-day hospital readmissions using Logistic Regression, SMOTE, GridSearchCV, and threshold optimization.
# 🏥 Hospital Readmission Prediction System  
### Predicting 30-Day Patient Readmission Risk using Machine Learning

---

## 📌 Project Overview

Hospital readmissions are a major challenge in modern healthcare systems. Predicting whether a patient will be readmitted within 30 days allows hospitals to take preventive action, improve patient outcomes, and optimize resource utilization.

This project develops a **machine learning-based predictive system** that identifies high-risk patients using clinical and administrative data.

The system follows a complete end-to-end machine learning pipeline, from data preprocessing and imbalance handling to hyperparameter tuning and threshold optimization.

---

## 🎯 Project Objective

The primary goal of this project is to build a reliable binary classification model that predicts:

> Whether a patient will be readmitted to the hospital within 30 days.

Target variable:

```
readmitted_within_30days
```

Class interpretation:

| Class | Meaning |
|------|---------|
| 0 | Patient NOT readmitted |
| 1 | Patient readmitted |

This system enables proactive clinical intervention and improves hospital decision-making.

---

## 🧠 Machine Learning Pipeline Overview

This project implements a complete production-level ML pipeline:

```
Data Collection
↓
Data Cleaning & Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Feature Engineering
↓
Handling Class Imbalance (SMOTE)
↓
Model Training & Comparison
↓
Hyperparameter Tuning (GridSearchCV)
↓
Threshold Optimization
↓
Final Model Selection
↓
Evaluation & Clinical Interpretation
```

---

## 📊 Dataset Characteristics

The dataset contains clinical and administrative patient features such as:

- Age
- Gender
- Admission type
- Length of hospital stay
- Number of medications
- Previous admissions
- Diagnosis codes
- Medical procedures

These features are used to predict readmission probability.

---

## ⚠️ Handling Class Imbalance using SMOTE

Healthcare datasets often suffer from class imbalance, where readmitted patients are fewer than non-readmitted patients.

To address this, SMOTE (Synthetic Minority Oversampling Technique) was applied **only to the training data**.

Benefits:

- Prevents model bias toward majority class
- Improves model ability to detect readmitted patients
- Ensures realistic evaluation on untouched test data
- Prevents data leakage

---

## 🤖 Models Implemented and Evaluated

The following machine learning models were trained and evaluated:

- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Gradient Boosting Classifier

Each model was evaluated using:

- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

## ⚙️ Hyperparameter Optimization using GridSearchCV

GridSearchCV was applied to optimize hyperparameters for tree-based models.

Parameters tuned included:

- Tree depth
- Number of estimators
- Minimum samples split
- Learning rate

Despite optimization, tree-based models did not outperform Logistic Regression due to the probabilistic structure of the dataset.

This indicates that the dataset exhibits primarily linear probabilistic relationships.

---

## 🎯 Threshold Optimization (Critical Step)

Unlike most beginner projects, this project implemented **probability threshold optimization**.

Logistic Regression outputs prediction probabilities.

Instead of using default threshold:

```
Default threshold = 0.50
```

Threshold was optimized using:

- Precision
- Recall
- F1 Score

This allowed achieving the best balance between:

- Detecting high-risk patients (Recall)
- Preventing excessive false alarms (Precision)

This step significantly improved clinical reliability.

---

## 🏆 Final Model Selection: Logistic Regression

Logistic Regression was selected as the final model because it provided:

✔ Best balance of Precision and Recall  
✔ Stable performance across test data  
✔ Strong generalization capability  
✔ Probabilistic prediction flexibility  
✔ Clinical interpretability  

More complex models did not provide superior performance after optimization.

This confirms Logistic Regression as the most reliable model for this dataset.

---

## 📈 Final Model Performance

Key performance characteristics:

- High recall for readmitted patients
- Stable probabilistic predictions
- Balanced precision-recall trade-off
- Reliable real-world applicability

The model effectively identifies high-risk patients.

---

## 🏥 Business and Clinical Impact

This system provides significant healthcare value:

### Clinical Benefits

- Early identification of high-risk patients
- Improved patient monitoring
- Better discharge planning
- Improved patient safety

### Operational Benefits

- Reduced hospital readmission rates
- Optimized hospital resource allocation
- Reduced healthcare costs
- Improved hospital efficiency

This enables proactive healthcare management.

---

## 🔬 Technical Highlights

This project demonstrates advanced machine learning engineering practices:

- Complete end-to-end ML pipeline
- Proper handling of imbalanced clinical data
- SMOTE implementation without data leakage
- Multi-model training and comparison
- Hyperparameter optimization using GridSearchCV
- Threshold optimization using probabilistic analysis
- Clinical-focused model selection

This reflects real-world healthcare AI development workflows.

---

## 🧾 Project Structure

```
readmission-ml/
│
├── dataset/
│   └── readmission_data.csv
│
├── notebooks/
│   └── readmission.ipynb
│
├── model/
│   └── trained_model.pkl
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- imbalanced-learn (SMOTE)

---

## 🚀 Key Learning Outcomes

This project demonstrates expertise in:

- Handling imbalanced healthcare datasets
- Model comparison and selection
- Hyperparameter tuning
- Threshold optimization
- Clinical ML interpretation
- Production-level ML pipeline design

---

## 🎯 Final Conclusion

This project successfully developed a machine learning system capable of predicting hospital readmissions with strong reliability and clinical relevance.

Logistic Regression was selected as the final model due to its superior stability, interpretability, and probabilistic prediction capability.

The system is suitable for integration into real-world hospital decision support systems.

---

## 👨‍💻 Author

Mohammed Panchla

---

## ⭐ If you found this project useful, consider giving it a star!
