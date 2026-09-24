# Fraud Intelligent System

A Django-based machine learning application for detecting fraudulent financial transactions and analyzing fraud patterns through an interactive dashboard.

## Project Overview

The Fraud Intelligent System uses machine learning to identify potentially fraudulent transactions. It provides prediction, data analysis, transaction search, and dashboard features through a Django web application.

## Technologies Used

* Python
* Django
* Machine Learning
* Random Forest
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* HTML
* CSS
* SQLite

## Machine Learning

The project uses a Random Forest classification model to detect fraudulent transactions.

The dataset contains transaction features including:

* Time
* V1–V28
* Amount
* Class

The `Class` column represents the transaction category, where fraudulent and normal transactions are identified.

## Features

* Fraud transaction prediction
* CSV transaction analysis
* Interactive dashboard
* Transaction search
* Fraud and normal transaction statistics
* Machine learning model integration
* Data visualization
* Result analysis

## Project Structure

fraud-intelligent-system/
detector/ - Django application
fraud_system/ - Django project configuration
gis/ - CSS, JavaScript and image files
model/ - Trained Random Forest model files
templates/ - HTML templates
manage.py - Django management script
requirements.txt - Python dependencies
analyze_data.py - Dataset analysis
check_dataset.py - Dataset validation
compare_models.py - Model comparison
create_test_csv.py - Test CSV generation

## Model

The trained Random Forest model and preprocessing files are stored in the `model` directory.

## Purpose

This project was developed as a machine learning and Django project to demonstrate fraud detection, data analysis, model evaluation, and web application development.

## Author

**Dhanusri Gangada**

GitHub: https://github.com/dhanusrigangada9-glitch
