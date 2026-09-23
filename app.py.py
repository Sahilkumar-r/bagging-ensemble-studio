import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

st.set_page_config(page_title="Bagging Ensemble Studio", layout="wide")

st.title("Bagging Ensemble Studio")
st.markdown("Upload a clean dataset, configure your base estimator, and train a bagging ensemble.")

# --- 1. File Upload ---
uploaded_file = st.file_uploader("Upload preprocessed dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head(), use_container_width=True)

    # --- 2. Data Configuration ---
    st.sidebar.header("1. Data Configuration")
    task_type = st.sidebar.selectbox("Task Type", ["Classification", "Regression"])
    
    target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns", 
                                          options=[col for col in df.columns if col != target_col], 
                                          default=[col for col in df.columns if col != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # --- 3. Base Algorithm Configuration ---
        st.sidebar.header("2. Base Algorithm Configuration")
        
        if task_type == "Classification":
            base_algo = st.sidebar.selectbox("Base Algorithm", ["Decision Tree", "Logistic Regression", "SVC"])
        else:
            base_algo = st.sidebar.selectbox("Base Algorithm", ["Decision Tree", "Linear Regression", "Ridge", "Lasso", "SVR"])

        base_estimator = None

        # Hyperparameters based on selection
        if base_algo == "Decision Tree":
            max_depth = st.sidebar.slider("Max Depth", 1, 50, 10, key="dt_depth")
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2, key="dt_split")
            if task_type == "Classification":
                base_estimator = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            else:
                base_estimator = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

        elif base_algo == "Logistic Regression":
            C = st.sidebar.number_input("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0, key="lr_c")
            penalty = st.sidebar.selectbox("Penalty", ["l2", "None"], key="lr_pen")
            pen = None if penalty == "None" else penalty
            base_estimator = LogisticRegression(C=C, penalty=pen, solver='lbfgs', max_iter=1000)

        elif base_algo == "Linear Regression":
            base_estimator = LinearRegression()
            
        elif base_algo in ["Ridge", "Lasso"]:
            alpha = st.sidebar.number_input("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, key="reg_alpha")
            if base_algo == "Ridge":
                base_estimator = Ridge(alpha=alpha)
            else:
                base_estimator = Lasso(alpha=alpha)

        elif base_algo in ["SVC", "SVR"]:
            C = st.sidebar.number_input("Regularization Parameter (C)", 0.1, 10.0, 1.0, key="svc_c")
            kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"], key="svc_kernel")
            if task_type == "Classification":
                base_estimator = SVC(C=C, kernel=kernel)
            else:
                base_estimator = SVR(C=C, kernel=kernel)

        # --- 4. Bagging Configuration ---
        st.sidebar.header("3. Bagging Configuration")
        n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 50, step=10)
        max_samples = st.sidebar.slider("Max Samples (Ratio)", 0.1, 1.0, 1.0, step=0.1)
        max_features = st.sidebar.slider("Max Features (Ratio)", 0.1, 1.0, 1.0, step=0.1)
        bootstrap = st.sidebar.checkbox("Bootstrap Samples", value=True)

        # --- 5. Validation Configuration ---
        st.sidebar.header("4. Validation Strategy")
        val_method = st.sidebar.radio("Method", ["Train-Test Split", "K-Fold Cross Validation"])
        
        if val_method == "Train-Test Split":
            test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, step=0.05)
        else:
            k_folds = st.sidebar.slider("Number of Folds (K)", 2, 10, 5)

        # --- 6. Execution ---
        if st.button("Train & Evaluate Ensemble", type="primary"):
            with st.spinner("Training model..."):
                
                # Initialize Bagging Model
                if task_type == "Classification":
                    ensemble = BaggingClassifier(
                        estimator=base_estimator,
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        n_jobs=-1,
                        random_state=42
                    )
                else:
                    ensemble = BaggingRegressor(
                        estimator=base_estimator,
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        n_jobs=-1,
                        random_state=42
                    )

                st.subheader("Model Evaluation Results")

                if val_method == "Train-Test Split":
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    ensemble.fit(X_train, y_train)
                    y_pred = ensemble.predict(X_test)
                    
                    col1, col2 = st.columns(2)
                    if task_type == "Classification":
                        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                        col2.metric("F1 Score (Weighted)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
                    else:
                        col1.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                        col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.4f}")

                else: # K-Fold
                    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                    
                    if task_type == "Classification":
                        scores = cross_val_score(ensemble, X, y, cv=kf, scoring='accuracy', n_jobs=-1)
                        metric_name = "Accuracy"
                    else:
                        scores = cross_val_score(ensemble, X, y, cv=kf, scoring='r2', n_jobs=-1)
                        metric_name = "R² Score"
                        
                    col1, col2 = st.columns(2)
                    col1.metric(f"Mean {metric_name} (across {k_folds} folds)", f"{scores.mean():.4f}")
                    col2.metric("Standard Deviation", f"{scores.std():.4f}")
                    
                    st.write(f"**Fold-by-fold {metric_name} scores:**")
                    st.bar_chart(scores)