# 🏥 AI Diabetes Diagnostic Dashboard

An end-to-end Machine Learning web application for predicting diabetes risk.

---

## 📝 Overview

This project takes 8 basic clinical metrics (such as Glucose, BMI, Age, and Insulin) and uses an optimized Random Forest machine learning model to predict the likelihood of a patient having diabetes. It features a fully interactive web dashboard that generates styled medical report cards.

---

## 📑 Table of Contents

- [✨ Key Features](#key-features)
- [🛠️ Tech Stack](#tech-stack)
- [📊 Model Performance](#model-performance)
- [📂 Project Structure](#project-structure)
- [🚀 Getting Started](#getting-started)
- [📸 Screenshots](#screenshots)
- [📜 Acknowledgements](#acknowledgements)

- [🔗 Live Demo](#live-demo)

---

## 🔗 Live Demo

[![Open in Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-orange?logo=huggingface&label=Live%20Demo)](https://huggingface.co/spaces/naimul3103/diabetes-prediction-system-for-women)

Try the live application on Hugging Face Spaces: https://huggingface.co/spaces/naimul3103/diabetes-prediction-system-for-women

---

## ✨ Key Features

- **Advanced Data Processing:** Handles biologically impossible zeros (e.g., zero blood pressure), utilizes median imputation, caps outliers via the IQR method, and engineers robust clinical features (BMI Category & Glucose-to-Insulin Ratio).
- **Robust Pipeline:** Uses `sklearn.pipeline` to strictly prevent data leakage between train and test sets during scaling and imputation.
- **Hyperparameter Tuned:** The Random Forest algorithm was rigorously optimized using `GridSearchCV` and evaluated with Stratified 10-Fold Cross-Validation.
- **Interactive UI:** A highly intuitive, stylized Gradio dashboard that instantly analyzes patient profiles, complete with risk probability bars and dynamic clinical advice.

---

## 🛠️ Tech Stack

| Category           | Technologies Used            |
| ------------------ | ---------------------------- |
| Language           | Python 3.13                  |
| Data Processing    | Pandas, NumPy                |
| Machine Learning   | Scikit-Learn (Random Forest) |
| Data Visualization | Matplotlib, Seaborn          |
| Frontend/Web App   | Gradio, HTML/CSS             |

---

## 📊 Model Performance

The model was validated using 10-Fold Stratified Cross-Validation. The final test set metrics achieved are:

| Metric   | Score  | Note                                              |
| -------- | ------ | ------------------------------------------------- |
| Accuracy | ~76.0% | Overall correctness of the model.                 |
| ROC-AUC  | ~0.831 | Excellent ability to distinguish between classes. |

🏆 **Top Clinical Predictors Identified:** Glucose levels, BMI, and Age.

---

## 📂 Project Structure

```
diabetes-prediction-system/
│
├── app.py                # Gradio web application & UI layout
├── final_solution_main.py # Data cleaning, EDA, Pipeline building, & Training
├── diabetes.csv           # Dataset (Kaggle / UCI Machine Learning)
├── best_model.pkl         # Serialized optimal ML pipeline (Pickle)
├── .gitignore             # Ignored files for version control
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine. **We highly recommend using Anaconda to prevent dependency conflicts.**

### 1. Clone the repository

```bash
git clone https://github.com/your-username/diabetes-prediction-system.git
cd diabetes-prediction-system
```

### 2. Environment Setup

Create and activate an isolated Conda environment:

```bash
conda create --name diabetes_env python=3.13 -y
conda activate diabetes_env
```

### 3. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn gradio
```

### 4. Run the Application

You have two options depending on what you want to do:

#### Option A: Launch the Web Dashboard

Use this if you just want to interact with the pre-trained model.

```bash
python app.py
```

Open the local URL (usually http://127.0.0.1:7860) provided in your terminal.

Or, use the hosted demo on Hugging Face Spaces (no local setup required):

https://huggingface.co/spaces/naimul3103/diabetes-prediction-system-for-women

#### Option B: Re-train the Model

Use this if you want to run the EDA, view visual plots, and train a new `best_model.pkl`.

```bash
python final_solution_main.py
```

---

## 📸 Screenshots

_(Replace this section with actual screenshots of your new Gradio Dashboard once you push your code to GitHub)_

🖼️ **Insert Screenshot Here:** Image of the main input dashboard.

🖼️ **Insert Screenshot Here:** Image of the green/red HTML medical report card output.

---

## 📜 Acknowledgements

- **Dataset:** Pima Indians Diabetes Database originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
- **Disclaimer:** This dashboard and associated machine learning models are built for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
