# Fraud Intelligent System

A Django-based machine learning web application for detecting fraudulent financial transactions, analyzing transaction data, searching transactions, and viewing fraud analytics through an interactive dashboard.

## Project Overview

The **Fraud Intelligent System** is a machine learning and Django web application developed to identify potentially fraudulent financial transactions.

The system integrates a trained **Random Forest classification model** with a Django web interface. Users can analyze transaction data, make fraud predictions, search transactions, and view statistical insights through the dashboard.

## Technologies Used

* Python
* Django
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Random Forest
* HTML
* CSS
* JavaScript
* SQLite

## Machine Learning

The application uses a **Random Forest Classifier** for fraud detection.

The transaction dataset contains **284,807 records** and **31 columns**.

### Input Features

* `Time`
* `V1` – `V28`
* `Amount`

### Target

* `Class`

The `Class` column identifies whether a transaction is normal or fraudulent.

* `0` – Normal transaction
* `1` – Fraudulent transaction

## Model Performance

The Random Forest model evaluated in the project achieved the following results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 99.95% |
| Precision | 90.59% |
| Recall    | 78.57% |
| F1 Score  | 84.15% |

### Confusion Matrix

|               | Predicted Normal | Predicted Fraud |
| ------------- | ---------------: | --------------: |
| Actual Normal |          284,300 |              15 |
| Actual Fraud  |               21 |             471 |

* True Negative (TN): 284,300
* False Positive (FP): 15
* False Negative (FN): 21
* True Positive (TP): 471

## Application Features

### 1. Fraud Prediction

Predict whether an individual transaction is potentially fraudulent using the trained Random Forest model.

### 2. CSV File Upload

Upload transaction data through the web application for fraud analysis and prediction.

### 3. Transaction Search

Search and examine transaction records through the application.

### 4. Analytics Dashboard

The dashboard provides fraud-related statistics and visualizations, including:

* Total transactions
* Fraudulent transactions
* Normal transactions
* Fraud rate
* Total transaction amount
* Fraud transaction amount
* Average transaction amount
* Peak fraud hour
* Model performance metrics
* Confusion matrix
* Data visualizations

### 5. Data Analysis

The project includes Python utility scripts for examining the dataset, validating data, comparing models, and creating test CSV files.

## Dashboard Statistics

Based on the project dataset, the dashboard displays:

| Statistic                       |          Value |
| ------------------------------- | -------------: |
| Total Transactions              |        284,807 |
| Fraudulent Transactions         |            492 |
| Normal Transactions             |        284,315 |
| Fraud Rate                      |          0.17% |
| Total Transaction Amount        | ₹25,162,590.01 |
| Fraud Transaction Amount        |     ₹60,127.97 |
| Average Transaction Amount      |         ₹88.35 |
| Peak Fraud Hour                 |          11:00 |
| Transactions at Peak Fraud Hour |             56 |

## Project Structure

```text
fraud-intelligent-system/
│
├── detector/
│   ├── README.md
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
│
├── fraud_system/
│   ├── README.md
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── gis/
│   ├── css/
│   ├── img/
│   ├── js/
│   └── .gitkeep
│
├── model/
│   ├── README.md
│   ├── metrics.pkl
│   ├── random_forest_model.pkl
│   └── random_forest_scaler.pkl
│
├── templates/
│   ├── .gitkeep
│   ├── base.html
│   ├── dashboard.html
│   ├── home.html
│   ├── predict.html
│   ├── search.html
│   ├── search_transactions.html
│   ├── transcations.html
│   └── upload.html
│
├── .gitignore
├── README.md
├── analyze_data.py
├── check_dataset.py
├── compare_models.py
├── create_test_csv.py
├── manage.py
└── requirements.txt
```

## Important Files

### Django Application

`detector/`

Contains the main application logic, forms, views, models, URLs, and tests.

### Django Project

`fraud_system/`

Contains the Django project configuration, settings, URL configuration, ASGI, and WSGI files.

### Templates

`templates/`

Contains the HTML pages used by the web application.

### Static Files

`gis/`

Contains the project's CSS, JavaScript, and image resources.

### Machine Learning Model

`model/`

Contains the trained Random Forest model, scaler, and saved evaluation metrics.

## Model Files

The `model` directory contains:

* `random_forest_model.pkl` – trained Random Forest classification model
* `random_forest_scaler.pkl` – preprocessing/scaling object
* `metrics.pkl` – saved model evaluation metrics

## Utility Scripts

### `analyze_data.py`

Used for analyzing the transaction dataset.

### `check_dataset.py`

Used for checking and validating the dataset.

### `compare_models.py`

Used for comparing machine learning model performance.

### `create_test_csv.py`

Used to create test CSV data for application testing.

## Dataset

The original dataset contains:

* **284,807 transactions**
* **31 columns**
* **492 fraudulent transactions**
* **284,315 normal transactions**

The original `fraud_data.csv` file is **not included in this GitHub repository because of its large file size**.

The application expects the dataset at:

```text
Dataset/fraud_data.csv
```

## Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/dhanusrigangada9-glitch/fraud-intelligent-system.git
```

### 2. Open the Project

Open the cloned project folder in VS Code.

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the Dataset

Place the required dataset in:

```text
Dataset/fraud_data.csv
```

### 7. Run Django

```bash
python manage.py runserver
```

### 8. Open the Application

```text
http://127.0.0.1:8000/
```

## Application Pages

### Home

```text
http://127.0.0.1:8000/
```

### Fraud Prediction

```text
http://127.0.0.1:8000/predict/
```

### Upload File

```text
http://127.0.0.1:8000/upload/
```

### Search Transactions

```text
http://127.0.0.1:8000/search/
```

### Dashboard

```text
http://127.0.0.1:8000/dashboard/
```

## Application Workflow

```text
Transaction Data
       ↓
Data Processing
       ↓
Feature Scaling
       ↓
Random Forest Model
       ↓
Fraud Prediction
       ↓
Result Analysis
       ↓
Dashboard Visualization
```

## Project Purpose

This project demonstrates the practical integration of:

* Machine learning
* Fraud detection
* Data analysis
* Data visualization
* Django web development
* Model evaluation
* CSV data processing

It was developed as a portfolio project to demonstrate practical skills in **Data Science, Machine Learning, Python, and Django**.

## Author

**Dhanusri Gangada**

GitHub: `dhanusrigangada9-glitch`
