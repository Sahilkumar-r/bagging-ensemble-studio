# Bagging Ensemble Studio 🚀

A robust, interactive Streamlit web application that allows users to build, configure, and evaluate Bagging Ensembles on their own datasets without writing any code.

## 🌟 Features

*   **Custom Data Upload:** Upload any clean, preprocessed CSV dataset.
*   **Dynamic Configuration:** Automatically detects columns to let you easily select your target and features.
*   **Task Versatility:** Supports both Classification and Regression tasks.
*   **Base Estimator Selection:** Choose and tune hyperparameters for various base models:
    *   Decision Trees (Classifier/Regressor)
    *   Linear/Logistic Regression
    *   Ridge / Lasso
    *   Support Vector Machines (SVC/SVR)
*   **Bagging Parameters:** Fine-tune the ensemble by adjusting the number of estimators, max samples, max features, and bootstrapping.
*   **Validation Strategies:** Evaluate your model using either a standard Train-Test Split or K-Fold Cross Validation.

## 🛠️ Installation

First, clone this repository to your local machine and navigate into the directory:
```bash
git clone https://github.com/yourusername/bagging-ensemble-studio.git
cd bagging-ensemble-studio
```

Choose one of the following methods to install the dependencies:

### Option 1: Standard Installation using `pip`
It is recommended to set up a virtual environment first, then use the provided `requirements.txt` file:
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Ultra-fast Installation using `uv`
If you use [uv](https://github.com/astral-sh/uv) (an extremely fast Python package installer and resolver), you can initialize a project environment and add the dependencies directly:
```bash
# Initialize a uv project
uv init

# Add the required packages
uv add streamlit pandas numpy scikit-learn
```
*(Note: You can also simply run `uv pip install -r requirements.txt` to use the existing file with uv).*

## 🚀 Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This will open a new tab in your default web browser where you can interact with the Bagging Ensemble Studio.

## 📝 How to use the App
1. **Upload Data:** Use the file uploader to provide your clean `.csv` file.
2. **Configure Data:** Select if your task is Classification or Regression, then define your Target variable and Feature columns.
3. **Set Base Algorithm:** Choose your underlying model and tweak its specific hyperparameters (like Max Depth, C value, or Kernel).
4. **Set Bagging Config:** Adjust how the bagging meta-estimator behaves.
5. **Evaluate:** Choose your validation method and click "Train & Evaluate Ensemble" to see your metrics!
