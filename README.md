# 🏥 Medical Cost Estimation Using Machine Learning Regression

## 📌 Project Overview

This project predicts **medical insurance costs** based on a person's demographic and health-related information using **Machine Learning Regression**. The application is built with **Python**, **Scikit-learn**, and **Streamlit**, providing an easy-to-use web interface for estimating medical expenses.

---

## 🚀 Features

* Predicts medical insurance costs instantly.
* Interactive web interface using Streamlit.
* Trains a Machine Learning Regression model automatically.
* Displays model performance (R² Score).
* Shows feature importance visualization.
* Simple, single-file implementation.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```text
Medical-Cost-Estimator/
│── app.py
│── insurance.csv
│── requirements.txt
└── README.md
```

---

## 📊 Dataset

This project uses the **Medical Cost Personal Dataset** from Kaggle.

**Dataset Link:**
https://www.kaggle.com/datasets/mirichoi0218/insurance

### Dataset Features

| Feature  | Description                              |
| -------- | ---------------------------------------- |
| age      | Age of the person                        |
| sex      | Gender                                   |
| bmi      | Body Mass Index                          |
| children | Number of dependent children             |
| smoker   | Smoking status                           |
| region   | Residential region                       |
| charges  | Medical insurance cost (Target Variable) |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Medical-Cost-Estimator.git
```

Navigate to the project folder:

```bash
cd Medical-Cost-Estimator
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

## 📈 Machine Learning Workflow

1. Load the dataset
2. Preprocess the data
3. Encode categorical features
4. Split the dataset into training and testing sets
5. Train a Regression model
6. Evaluate model performance
7. Predict medical insurance costs
8. Display results in the Streamlit application

---

## 📸 Application Preview

The application allows users to:

* Enter age
* Select gender
* Enter BMI
* Select number of children
* Choose smoking status
* Select region
* Predict estimated medical insurance cost

---

## 📊 Model Evaluation

The project evaluates model performance using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

---

## 🎯 Future Improvements

* Compare multiple regression algorithms
* Add model selection option
* Generate downloadable PDF reports
* Deploy on Streamlit Community Cloud
* Improve prediction accuracy with feature engineering
* Add data visualization dashboards

---

## 👨‍💻 Author

**Krishna Mahto**

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it for educational and personal purposes.
