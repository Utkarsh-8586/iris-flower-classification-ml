# 🌸 Iris Flower Classification

An end-to-end machine learning project that classifies Iris flowers into
**Setosa**, **Versicolor**, or **Virginica** based on sepal and petal
measurements — trained, evaluated, and deployed as an interactive Streamlit app.

## Project Structure

```
iris-classifier/
├── train_model.py               # Full training pipeline (EDA, training, evaluation, saving)
├── app.py                       # Streamlit web app for live predictions
├── requirements.txt
├── model_comparison_results.csv # Metrics for all 5 models
├── data/
│   └── iris.csv                 # Exported dataset
├── models/
│   ├── best_model.pkl           # Trained best-performing model
│   ├── scaler.pkl               # Fitted StandardScaler
│   └── metadata.json            # Best model name, score, feature/target names
└── visuals/                     # All EDA and evaluation plots (PNG)
```

## Dataset

The **Iris dataset** (Fisher, 1936) — a classic, well-balanced 150-sample
dataset with 4 numeric features:

| Feature | Description |
|---|---|
| sepal length (cm) | Length of the sepal |
| sepal width (cm)  | Width of the sepal |
| petal length (cm) | Length of the petal |
| petal width (cm)  | Width of the petal |

Target: `species` — 50 samples each of Setosa, Versicolor, Virginica.

## Pipeline

1. **Data loading** — via `sklearn.datasets.load_iris`, exported to CSV.
2. **EDA** — summary stats, class balance, pairplot, correlation heatmap,
   boxplots (outlier/spread check) — saved to `visuals/`.
3. **Preprocessing** — `StandardScaler` feature scaling, 80/20 stratified
   train-test split.
4. **Model training** — 5 classifiers trained on the same split:
   - K-Nearest Neighbors
   - Decision Tree
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
5. **Evaluation** — Accuracy, macro Precision/Recall/F1, confusion matrix,
   and full classification report per model.
6. **Model selection** — best model chosen by macro F1-score, saved with
   Joblib alongside the scaler.
7. **Deployment** — Streamlit app with sliders for live predictions and a
   confidence bar chart.

## Results

On this run, models ranked as follows (see `model_comparison_results.csv`
for the exact numbers from your run — results can vary slightly with the
random train/test split):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| SVM | 0.967 | 0.970 | 0.967 | 0.967 |
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |
| KNN | 0.933 | 0.944 | 0.933 | 0.933 |
| Random Forest | 0.900 | 0.902 | 0.900 | 0.900 |

**Best model: SVM** — deployed in the app.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the training pipeline (generates plots + saves the model)
python train_model.py

# 3. Launch the web app
streamlit run app.py
```

## Why This Project Is Resume-Worthy

- Covers the **full ML lifecycle**: data → EDA → preprocessing → multi-model
  training → evaluation → selection → deployment.
- Uses **industry-standard tools** (Pandas, Scikit-Learn, Streamlit, Joblib).
- Demonstrates **comparative model evaluation**, not just one algorithm.
- Ships as a **working, interactive deployed app**, not just a notebook.
- Clean, modular code that's easy to explain in an interview.

## Possible Extensions (Talking Points for Interviews)

- Add hyperparameter tuning (`GridSearchCV`) for each model.
- Add k-fold cross-validation instead of a single train-test split.
- Deploy publicly via Streamlit Community Cloud.
- Add a "batch prediction" CSV upload feature to the app.
- Log experiments with MLflow.


👨‍💻 Author

UTKARSH CHAUHAN
B.Tech CSE | Lovely Professional University (LPU)

⭐ If you found this project useful, consider giving it a star on GitHub!
