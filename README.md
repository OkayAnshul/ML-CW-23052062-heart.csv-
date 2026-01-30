# Heart Disease Prediction using Machine Learning [ML-Assignment]

## 📌 Project Objective
The objective of this project is to implement and compare multiple Machine Learning classification algorithms on the Heart Disease dataset to predict whether a patient has heart disease or not.

This project demonstrates:
- Supervised Learning
- Classification algorithms
- Feature encoding
- Model evaluation
- Overfitting and pruning
- Model comparison

---

## 📂 Dataset
Dataset used: **UCI Heart Disease Dataset (from Kaggle)**  

The dataset contains 1025 samples and 13 input features such as:
- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- ECG results
- Maximum heart rate
- Exercise induced angina
- Oldpeak
- Slope
- Number of vessels colored
- Thalassemia  

Target variable:
- `0` → No heart disease  
- `1` → Heart disease present  

---

## 🧠 Machine Learning Algorithms Implemented
The following algorithms were implemented and compared:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Naive Bayes (GaussianNB)**
4. **Decision Tree (Unpruned)**
5. **Decision Tree (Pruned using max_depth)**

These algorithms are part of the course syllabus under:
- Regression and Classification
- Error Analysis
- Decision Tree and Pruning

---

## ⚙️ Data Preprocessing
- Categorical features were converted into numerical format using **One-Hot Encoding**
- Dataset was split into:
  - Training set (80%)
  - Testing set (20%)
- Feature Scaling (StandardScaler) was applied for:
  - Logistic Regression
  - KNN

---

## 📊 Evaluation Metrics
Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics are used to analyze performance and compare models objectively.

---

## 🧪 Results

| Algorithm | Accuracy |
|----------|----------|
| Logistic Regression | 82% |
| KNN | 87% |
| Naive Bayes | 78% |
| Decision Tree (Unpruned) | 99% (Overfitting) |
| Decision Tree (Pruned) | 83% |

A bar chart visualization was created to compare model performance.

---

## 📈 Observations
- KNN achieved the highest realistic accuracy (87%).
- Logistic Regression performance improved after feature scaling.
- Naive Bayes had lower accuracy but high recall for heart disease detection.
- Decision Tree without pruning overfits the dataset.
- Pruned Decision Tree gave more balanced and realistic results.

---

## 🧩 Concepts Demonstrated (As per Syllabus)
- Supervised Learning
- Classification
- Normalization and Standardization
- Error Analysis
- Bias-Variance Tradeoff
- Overfitting and Underfitting
- Decision Tree Pre-pruning
- Model Comparison

---

## 🗂 Project Structure
```
heart-disease-ml-project/
│
├── dataset/
│ └── heart.csv
│
├── notebooks/
│ ├── explore_data.ipynb
│ ├── logistic_regression.ipynb
│ ├── knn_model.ipynb
│ ├── naive_bayes.ipynb
│ ├── decision_tree.ipynb
│ └── model_comparison.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore

```
---

## 🛠 Tools and Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📦 Installation and Usage

Clone the repository:
```bash
[git clone https://github.com/OkayAnshul/ML-CW-23052062-heart.csv-.git]


Install dependencies:

pip install -r requirements.txt


Run notebooks:

jupyter notebook


Open notebooks inside the notebooks/ folder and execute them sequentially.

🎯 Conclusion

This project demonstrates how different Machine Learning algorithms perform on the same dataset. It also highlights the importance of preprocessing, model evaluation, and avoiding overfitting. Among all models tested, KNN performed best with 87% accuracy.

This project fulfills the requirements of implementing and comparing ML algorithms as per the syllabus.

👨‍🎓 Author

Anshul
Machine Learning Course Assignment

📜 License

This project is for educational purposes only.

