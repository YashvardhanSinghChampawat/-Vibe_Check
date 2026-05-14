# 🚀 Antigravity ML Intelligence Dashboard

Welcome to the **Antigravity ML Intelligence Dashboard**! This project is a comprehensive, futuristic, and highly interactive Machine Learning application built purely in Python using Streamlit. It allows users to upload data, train multiple state-of-the-art ML models, explore complex 3D relationships, and generate explainable insights—all wrapped in a deep space / antigravity UI aesthetic.

---

## ✨ Key Features

This dashboard is divided into 10 distinct, fully-functional sections:

1. **🏠 Home & Project Explanation**: A beginner-friendly introduction to Machine Learning, Random Forests, and the purpose of the dashboard. Features animated metric cards and quick-start links.
2. **📤 Data Input & Upload**: Upload any custom `.csv` file or load built-in datasets (including a dynamically generated "Antigravity Physics" synthetic dataset). Includes auto-cleaning and missing-value handling.
3. **📊 Data Exploration**: An interactive tabbed interface showing data previews, statistical summaries, missing value heatmaps, distributions, and an immersive 3D Correlation Cube.
4. **🤖 Model Training & Comparison**: Train up to 6 different models simultaneously (Random Forest, Logistic Regression, SVM, KNN, Decision Tree, XGBoost). View performance across an animated Radar Chart, Confusion Matrices, and ROC curves.
5. **🌐 3D Visualization Studio**: The visual highlight of the app! Powered by Plotly, this section renders:
   - 3D Scatter plots of data
   - 3D Decision Boundary Surfaces
   - 3D Feature Importance Forests
   - 3D Confusion Matrix Cubes
   - 3D t-SNE Manifold representations
6. **🔮 Live Prediction Panel**: A real-time prediction engine. Use sliders to input custom feature values and instantly receive model predictions with confidence intervals and probabilities.
7. **🧠 SHAP Explainability (XAI)**: Demystify the "black box" of Machine Learning. Uses `shap.TreeExplainer` to render SHAP Summary plots and Feature Importance Bar charts, explaining the *why* behind predictions.
8. **📈 Training Dynamics**: Visualizes learning curves and the bias-variance tradeoff to help understand model generalization.
9. **🎮 Gamification & Scorecard**: A dynamic achievement system that grades your model (from S to F) and awards badges (like "Accuracy Master" or "Speed Demon") based on performance and exploration.
10. **📝 Summary & Export**: Automatically generates a theoretical project summary and allows you to export your results as a **PDF Report**, **Excel Spreadsheet**, or **JSON Configuration**.

---

## 📦 Tech Stack

- **Frontend/Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Visualizations**: [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Explainable AI**: [SHAP](https://shap.readthedocs.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Exporting**: `reportlab` (PDF), `openpyxl` (Excel)

---

## 🛠️ Installation & Setup

1. **Clone the repository** (or download the files to your local machine).
2. **Install the dependencies**:
   ```bash
   pip install streamlit scikit-learn plotly pandas numpy matplotlib seaborn shap xgboost reportlab openpyxl
   ```
3. **Run the application**:
   Make sure you are in the project folder, then run:
   ```bash
   python -m streamlit run app.py
   ```
4. **View the Dashboard**: The application will automatically open in your default web browser at `http://localhost:8501`.

---

*“The goal is to turn data into information, and information into insight.”* — Carly Fiorina
