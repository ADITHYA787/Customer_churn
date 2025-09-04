# 📉 Customer Churn Prediction System

A robust, end-to-end machine learning pipeline for predicting customer churn using behavioral and demographic data. Built using XGBoost and Python, the model enables businesses to proactively identify at-risk users and take retention actions.

---

Features

- **Supervised ML Classification**: Uses XGBoost for high-performance binary classification.
- **Feature Engineering**: Encodes categorical and numerical data with preprocessing pipelines.
- **Churn Probability Scoring**: Returns likelihood of churn along with class label.
- **Feature Importance Analysis**: Provides insights into top features influencing churn.
- **Train-Test Evaluation**: Includes precision, recall, F1-score, and ROC curve evaluation.
- **Model Serialization**: Saves trained model using `joblib` for deployment or inference use.
- **Modular Codebase**: Clean structure for preprocessing, training, evaluation, and inference steps.
- **Optional Flask/FastAPI Ready**: Model can be integrated into API services for real-time usage.

---

Architecture

Components

- **Data Processing**: Cleans input, handles missing values, and encodes features.
- **Model Training**: Builds and evaluates the churn classifier.
- **Evaluation Module**: Reports key metrics and plots ROC/AUC curves.
- **Feature Importance**: Visualizes which features drive churn the most.
- **Model Export**: Stores trained `.pkl` file for inference.

---

Installation

 Clone the repository:
```bash
git clone [repository-url]
cd churn_prediction
```

 Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

Usage

 Run the notebook:
```bash
jupyter notebook churn_prediction.ipynb
```

 Export the model:
```python
import joblib
joblib.dump(model, "churn_model.pkl")
```

---

 Model Evaluation

- **Metrics Reported**:
  - Accuracy
  - Precision / Recall
  - F1-score
  - ROC-AUC

Feature Importance**:
  - Gain: Average performance gain of splits using the feature.
  - Weight: Frequency of feature used in decision trees.

 Example Output

```
Top 5 Features by Gain:
1. Monthly_Usage_Hours     (Gain: 0.215)
2. Plan_Type               (Gain: 0.190)
3. Tenure_Days             (Gain: 0.132)
4. Support_Tickets         (Gain: 0.120)
5. Discount_Used           (Gain: 0.101)
```

---

Business Impact

- **Reduced Revenue Leakage**: Proactively intervene with at-risk users via discounts or engagement.
- **Retention Campaign Targeting**: Allocate marketing resources to those most likely to churn.
- **Increased Customer Lifetime Value (CLV)**: Retain more customers and boost ROI.
- **Forecasting Stability**: Predict future churn for revenue planning and strategy.

---

Example Prediction Output

```json
{
  "input_user": {
    "Tenure_Days": 90,
    "Monthly_Usage_Hours": 15.5,
    "Support_Tickets": 3,
    "Discount_Used": 1
  },
  "prediction": "Churn",
  "churn_probability": 0.87
}
```

---

Configuration

Adjust the following settings within the notebook or config module:

- Categorical encoding strategy (LabelEncoder, OneHotEncoder)
- Test size split
- XGBoost hyperparameters
- Output file paths for model and plots

---

Testing

You can test the model by simulating new users or using a holdout test dataset. Sample prediction function included in the notebook.

Unit Test Coverage

- Data Preprocessing: Missing value handling, encoding validation
- Model Training: Fit test with sample inputs
- Evaluation: Confusion matrix, AUC-ROC test
- Inference: Single-row prediction logic test

---

Future Enhancements

- API Deployment with FastAPI or Flask
- Real-time churn prediction microservice
- Imbalanced classification handling (SMOTE)
- SHAP value explainability
- Integration with CRM systems (Salesforce, HubSpot)

---

Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request