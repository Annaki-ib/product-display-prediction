

# Product Display Prediction: Machine Learning in Retail

## Project Overview
In the competitive retail landscape, Product Display (extra shelving at aisle ends) is a prime driver of revenue. This project leverages Machine Learning to predict whether a product will be featured on a promotional display or remain on a standard shelf.

By analyzing historical sales volume, store turnovers, and retailer-specific behaviors, this tool serves as a decision-support system for Store Managers to optimize inventory and Brand Managers to audit promotional compliance.

---

## Business Value
* **Revenue Optimization:** Identify high-potential products that deserve premium floor space.
* **Compliance Auditing:** Verify if retailers are honoring "Display" agreements based on sales performance.
* **Automated Insights:** Move from manual shelf-checks to data-driven placement strategies.

---

## Project Structure

```text
├── data/
│   └── display_dataset.csv          # Cleaned and balanced retail dataset
├── Results/
│   └── Best_model/                  # Production artifacts
│       ├── best_model.pkl           # Serialized Random Forest Pipeline
│       └── plots/                   # Exported High-Res Visualizations
├── display_prediction.ipynb  # Full End-to-End Pipeline
├── README.md                        # Project Documentation
└── requirements.txt                 # Dependency list
```

---

## Methodology and Pipeline
The project follows a robust Data Science lifecycle:

1. **Data Engineering:**
   * Implementation of Scikit-Learn Pipelines for seamless data flow.

2. **Exploratory Data Analysis (EDA):**
   * Statistical analysis of sales distributions.
   * Correlation mapping to identify multi-collinearity between sales volume and value.

3. **Model Benchmarking:**
   * Tested 7 algorithms:DNN, Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Decision Trees.

4. **Model Selection:**
   * Evaluation based on ROC-AUC to ensure class discrimination in a retail context.

---

## Key Results
The **Random Forest Classifier** was selected as the champion model.

| Metric | Value |
| :--- | :--- |
| Accuracy | 84.2% |
| ROC-AUC | 0.91 |
| Primary Driver | Sales Volume (X1) |

### Visual Insights
* **Target Distribution:** Balanced dataset ensuring no model bias.
* **Feature Importance:** Sales Volume and Retailer ID are the top predictors.

### Model Robustness
* **Confusion Matrix:** Shows high precision for both "Display" and "No_Display" classes.
* **ROC Curve:** An AUC of 0.91 indicates excellent separation power.

---

## How to Use

### Installation
```bash
git clone https://github.com/Annaki-ib/product-display-prediction.git
cd product-display-prediction
pip install -r requirements.txt
```

### Inference
The model is stored as a serialized Joblib Pipeline. It includes both the preprocessor (scaling/encoding) and the classifier.

```python
import joblib

# Load the production pipeline
model = joblib.load('Results/Best_model/best_model.pkl')

# Run prediction
prediction = model.predict(new_data)
```

---

## Authors
* EL Mahdi ZOUI
* ANNAKI Ibtihal
