# Loan Approval Prediction System. 

A simple end-to-end machine learning project using Python, XGBoost and Streamlit. There is no backend or API server: Streamlit loads the trained model and makes predictions directly.

## Website Link :   
https://loan-approval-system-technexus-neqhjn5cvldr32ddx3meus.streamlit.app/


## Project structure

```text
Loan-Approval-System/
├── dataset/
│   └── loan_data_new.csv
├── model/
│   └── loan_approval_xgboost.pkl
├── training/
│   └── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 1. Install packages

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Add the dataset

Put `loan_data_new.csv` inside the `dataset` folder.

The CSV must contain these exact columns:

```text
Age
Gender
Education
Person Income
Employee Experience
Home Onwership
Loan Amount
Loan Intent
Loan interest Rate
Loan percentage
Credit History
Credit Score
Previous Loan
Loan Status
```

`Loan Status` must contain 1/0 or Approved/Rejected. The spelling `Home Onwership` is intentional to match the supplied dataset's column name.

## 3. Train the model

Run from the main project folder:

```bash
python training/train_model.py
```

The script trains an XGBoost pipeline and saves:

- `model/loan_approval_xgboost.pkl`
- `model/metrics.json`

`Loan interest Rate` is excluded from model inputs to avoid possible target leakage. The app uses the same preprocessing pipeline saved during training.

## 4. Run Streamlit

```bash
streamlit run app.py
```

Your browser will open the application.

## 5. Deploy

For Streamlit Community Cloud:

1. Push the project to a GitHub repository.
2. Ensure the trained model file is included in the repository or otherwise made available during deployment.
3. Open Streamlit Community Cloud.
4. Select your repository and set the main file path to `app.py`.
5. Deploy.

This project is for educational demonstration only. Its predictions are not actual lending decisions.
