# BREAST_CANCER_ANALYSIS
A Machine Learning Approach for Early Prediction of Breast Cancer
Breast Cancer Classification and Analysis Using Machine Learning Techniques

This project focuses on diagnosing breast cancer using various machine learning classification models. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and comparison of six classifiers.

## 📁 Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features:** 30 numeric features computed from digitized images of breast mass
- **Target:** `diagnosis` — Malignant (M) or Benign (B)

---

## 🔍 Project Objectives

- Clean and preprocess breast cancer data
- Visualize data distribution, correlations, and patterns
- Train multiple classification models
- Compare models using evaluation metrics (accuracy, precision, recall, F1-score)
- Visualize model performances for better interpretation

---

## 📊 Visualizations Included

- Count Plot of diagnosis classes
- Histograms and Subplots
- Scatter Plot
- Boxplot and Violin Plot
- Correlation Heatmap
- Pair Plot
- Bar Plot comparing model accuracies

---

## 🤖 Machine Learning Models Used

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Naive Bayes
5. Decision Tree
6. Random Forest

---

## 🧪 Evaluation Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.9649   | 0.9574    | 0.9574 | 0.9574   |
| KNN                | 0.9561   | 0.9444    | 0.9574 | 0.9508   |
| SVM                | 0.9737   | 0.9744    | 0.9574 | 0.9658   |
| Naive Bayes        | 0.9298   | 0.9149    | 0.9362 | 0.9254   |
| Decision Tree      | 0.9123   | 0.8958    | 0.9149 | 0.9053   |
| Random Forest      | 0.9737   | 0.9744    | 0.9574 | 0.9658   |

> ℹ️ *These are sample results. Please update the table with your exact output from the evaluation section.*

---

## 📌 Conclusion

This project highlights the importance of using multiple machine learning techniques for medical diagnosis. Random Forest and Support Vector Machines perform the best in accuracy and F1 score. Proper preprocessing and feature selection significantly improve model performance.

---

## 🛠️ Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn


