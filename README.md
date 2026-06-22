# Heart Disease Prediction Using Logistic Regression and Random Forest Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Internship](https://img.shields.io/badge/CodeAlpha-Internship-red)](https://codealpha.tech/)

> Heart Disease Prediction using Logistic Regression and Random Forest Classifier with Data Preprocessing, EDA, Visualization, Model Evaluation and Prediction | CodeAlpha Machine Learning Internship Project

---

## 📌 Project Overview

This project predicts whether a patient is likely to have **heart disease** based on clinical features such as age, cholesterol, blood pressure, and more. It uses two machine learning models:

- **Logistic Regression** — Baseline classification model
- **Random Forest Classifier** — High-accuracy ensemble model

---

## 📁 Repository Structure

```
codealpha_task1_Heart_Disease_Prediction/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── dataset/
│   └── heart.csv
│
├── source_code/
│   ├── heart_disease_prediction.py
│   └── heart_disease_prediction.ipynb
│
├── report/
│   └── Heart_Disease_Prediction_Report.pdf
│
├── images/
│   ├── age_distribution_histogram.png
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── sample_prediction_1.png
│   └── sample_prediction_2.png
│
└── outputs/
    ├── accuracy_output.txt
    ├── classification_report.txt
    └── sample_predictions.txt
```

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository / Kaggle Heart Disease Dataset
- **File**: `dataset/heart.csv`
- **Features**: 13 clinical attributes (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target**: `0` = No Heart Disease, `1` = Heart Disease Present

---

## 🔬 Workflow

1. **Data Loading & Exploration** — Load CSV, shape, info, describe
2. **Exploratory Data Analysis (EDA)** — Distribution plots, heatmaps
3. **Data Preprocessing** — Handle missing values, feature scaling
4. **Model Training** — Logistic Regression & Random Forest Classifier
5. **Model Evaluation** — Accuracy, Confusion Matrix, Classification Report
6. **Sample Predictions** — Predict for new patient data

---

## 📈 Model Performance

| Model                    | Accuracy  |
|--------------------------|-----------|
| Random Forest Classifier | **98.53%** |
| Logistic Regression      | 79.50%    |

---

## 🖼️ Visualizations

| Plot | Description |
|------|-------------|
| `age_distribution_histogram.png` | Age distribution of patients |
| `target_distribution.png` | Heart disease class distribution |
| `correlation_heatmap.png` | Feature correlation heatmap |
| `confusion_matrix.png` | Model confusion matrix |
| `sample_prediction_1.png` | Sample prediction output 1 |
| `sample_prediction_2.png` | Sample prediction output 2 |

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/addadugurudurga2024-lang/codealpha_task1_Heart_Disease_Prediction.git
cd codealpha_task1_Heart_Disease_Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Python script
python source_code/heart_disease_prediction.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🏷️ Topics

`python` `machine-learning` `logistic-regression` `random-forest` `heart-disease-prediction` `classification` `scikit-learn` `data-science` `codealpha` `internship-project`

---

## 👨‍💻 Author

**A D S ABHISHEK**
- GitHub: [@addadugurudurga2024-lang](https://github.com/addadugurudurga2024-lang)
- Internship: CodeAlpha Machine Learning Intern

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [CodeAlpha](https://codealpha.tech/) for the internship opportunity
- UCI Machine Learning Repository for the dataset
- scikit-learn community
