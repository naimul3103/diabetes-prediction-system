🏥 AI Diabetes Diagnostic Dashboard

An end-to-end Machine Learning project that predicts the likelihood of diabetes in patients based on medical diagnostic measurements. This project utilizes the Pima Indians Diabetes Database, processes the data, trains a robust Random Forest Classifier, and serves the results through a modern, interactive Gradio web interface.

📋 Table of Contents

Project Overview

Project Structure

Model Architecture & Performance

Installation & Setup

Usage

License & Acknowledgements

🎯 Project Overview

This system takes in 8 biological metrics (such as Glucose, BMI, Age, Insulin, etc.) and generates engineered features (like BMI Category and Glucose-to-Insulin Ratio) to predict diabetes risk.

Key Features:

Comprehensive EDA & Preprocessing: Handles biologically impossible zeros (e.g., zero blood pressure), uses median imputation, clips outliers via the IQR method, and engineers new clinical features.

Pipeline Integration: Utilizes sklearn.pipeline to prevent data leakage during scaling and imputation.

Robust Training: Optimized a Random Forest model using GridSearchCV and Stratified 10-Fold Cross-Validation.

Interactive UI: A user-friendly dashboard built with Gradio to instantly analyze patient profiles and generate a styled HTML medical report card.

📂 Project Structure

.
├── app.py # Gradio web interface script
├── final_solution_main.py # Data processing, training, and evaluation script
├── diabetes.csv # Dataset (Kaggle / UCI Machine Learning)
├── best_model.pkl # Serialized optimal machine learning pipeline
├── README.md # Project documentation
└── .gitignore # Ignored files for git version control

🧠 Model Architecture & Performance

Algorithm: Random Forest Classifier (Tuned via GridSearch)

Validation: 10-Fold Stratified Cross-Validation

Final Metrics (Test Set):

Accuracy: ~76.0%

ROC-AUC: ~0.831

Top Predictors: Glucose, BMI, and Age were found to be the most critical features in predicting diabetes risk.

🚀 Installation & Setup

We recommend using Anaconda (conda) to manage the virtual environment to prevent dependency conflicts.

1. Clone the repository:

git clone [https://github.com/your-username/diabetes-prediction-system.git](https://github.com/your-username/diabetes-prediction-system.git)
cd diabetes-prediction-system

2. Create and activate a Conda environment:

conda create --name diabetes_env python=3.13
conda activate diabetes_env

3. Install required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn gradio

💻 Usage

Option 1: Train the Model
If you want to re-run the preprocessing, view the analytics, and retrain the model, run:

python final_solution_main.py

(This will output metrics to the console, display plots, and overwrite best_model.pkl if saving logic is included).

Option 2: Launch the Web Dashboard
To launch the interactive Gradio UI using the pre-trained best_model.pkl, simply run:

python app.py

The terminal will output a local URL (usually http://127.0.0.1:7860).

Open the URL in your web browser.

Input patient vitals or click the quick-test examples to see the AI diagnostic report card.

📜 License & Acknowledgements

Dataset: Pima Indians Diabetes Database provided by the UCI Machine Learning Repository.

Disclaimer: The dashboard is built for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
