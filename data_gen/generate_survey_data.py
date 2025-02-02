import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
num_patients = 2000
num_visits = 5000
num_attempts = 8000
num_responses = 5000

# Generate Patient Table
patients = []
for i in range(1, num_patients + 1):
    age = (
        np.random.randint(18, 85) if np.random.rand() > 0.1 else None
    )  # 10% missing age
    gender = random.choice(["Male", "Female", "Non-Binary"])
    location = random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Miami"])
    insurance = random.choice(["Private", "Medicare", "Medicaid", "Uninsured"])
    patients.append([i, age, gender, location, insurance])

df_patients = pd.DataFrame(
    patients, columns=["Patient_ID", "Age", "Gender", "Location", "Insurance_Type"]
)

# Generate Visit Table
visits = []
for i in range(1, num_visits + 1):
    patient_id = np.random.randint(1, num_patients + 1)
    visit_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 180))
    doctor_id = np.random.randint(1000, 1100)
    reason = random.choice(
        ["Checkup", "Emergency", "Follow-up", "Surgery", "Consultation"]
    )
    outcome = random.choice(["Discharged", "Admitted", "Follow-up Needed"])
    cost = round(np.random.uniform(100, 5000), 2)
    visits.append([i, patient_id, visit_date, doctor_id, reason, outcome, cost])

df_visits = pd.DataFrame(
    visits,
    columns=[
        "Visit_ID",
        "Patient_ID",
        "Visit_Date",
        "Primary_Dr_ID",
        "Visit_Reason",
        "Visit_Outcome",
        "Visit_Cost",
    ],
)

# Generate Survey Attempts Table
methods = ["SMS", "Email", "Phone"]
survey_attempts = []
for i in range(1, num_attempts + 1):
    patient_id = np.random.randint(1, num_patients + 1)
    visit_id = (
        np.random.randint(1, num_visits + 1) if np.random.rand() > 0.05 else None
    )  # 5% have no matching visit
    method = np.random.choice(methods, p=[0.4, 0.4, 0.2])
    attempt_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 180))
    # Get patient age
    age = df_patients.loc[df_patients["Patient_ID"] == patient_id, "Age"].values[0]

    # Adjust success rates based on age
    if age is not None:
        if age >= 60:
            success_rates = {"SMS": 0.1, "Email": 0.1, "Phone": 0.9}  # Favor phone
        elif age < 25:
            success_rates = {
                "SMS": 0.8,
                "Email": 0.5,
                "Phone": 0.1,
            }  # Favor SMS & Email
        else:
            success_rates = {"SMS": 0.5, "Email": 0.6, "Phone": 0.4}  # Default rates
    else:
        success_rates = {
            "SMS": 0.5,
            "Email": 0.6,
            "Phone": 0.4,
        }  # Default if age is missing

    success = np.random.rand() < success_rates[method]

    # Introduce duplicate attempts (~2% of records)
    if np.random.rand() < 0.02:
        survey_attempts.append([i, patient_id, visit_id, method, attempt_date, success])

    survey_attempts.append([i, patient_id, visit_id, method, attempt_date, success])

df_survey_attempts = pd.DataFrame(
    survey_attempts,
    columns=[
        "Attempt_ID",
        "Patient_ID",
        "Visit_ID",
        "Survey_Method",
        "Attempt_Date",
        "Attempt_Success",
    ],
)

# Generate Survey Responses Table (only for successful attempts)
survey_responses = []
for i in range(1, num_responses + 1):
    attempt_id = np.random.choice(
        df_survey_attempts[df_survey_attempts["Attempt_Success"] == True][
            "Attempt_ID"
        ].values
    )
    patient_id = df_survey_attempts[df_survey_attempts["Attempt_ID"] == attempt_id][
        "Patient_ID"
    ].values[0]
    visit_id = df_survey_attempts[df_survey_attempts["Attempt_ID"] == attempt_id][
        "Visit_ID"
    ].values[0]
    response_date = pd.to_datetime(
        df_survey_attempts[df_survey_attempts["Attempt_ID"] == attempt_id][
            "Attempt_Date"
        ].values[0]
    ) + timedelta(days=np.random.randint(2, 5))
    score = (
        np.random.randint(3, 6) if np.random.rand() > 0.1 else np.random.randint(1, 3)
    )  # 10% bad scores
    comments = (
        random.choice(
            [
                "Great service",
                "Average experience",
                "Would not recommend",
                "Excellent!",
                "Not happy",
            ]
        )
        if np.random.rand() > 0.3
        else None
    )

    survey_responses.append(
        [i, attempt_id, patient_id, visit_id, response_date, score, comments]
    )

df_survey_responses = pd.DataFrame(
    survey_responses,
    columns=[
        "Response_ID",
        "Attempt_ID",
        "Patient_ID",
        "Visit_ID",
        "Response_Date",
        "Satisfaction_Score",
        "Comments",
    ],
)

# Generate Survey Cost Table (external data for AI)
survey_costs = [
    ["Survey Software", round(np.random.uniform(5000, 20000), 2)],
    ["Survey Analysis", round(np.random.uniform(2000, 10000), 2)],
    [
        "SMS Costs",
        round(
            len(df_survey_attempts[df_survey_attempts["Survey_Method"] == "SMS"])
            * 0.02,
            2,
        ),
    ],
    [
        "Email Costs",
        round(
            len(df_survey_attempts[df_survey_attempts["Survey_Method"] == "Email"])
            * 0.01,
            2,
        ),
    ],
    [
        "Phone Costs",
        round(
            len(df_survey_attempts[df_survey_attempts["Survey_Method"] == "Phone"])
            * 0.04,
            2,
        ),
    ],
]

df_survey_costs = pd.DataFrame(survey_costs, columns=["Cost_Category", "Cost_Amount"])

# Generate Denormalized Survey Analytics Table
survey_analytics = (
    df_survey_attempts.merge(df_patients, on="Patient_ID", how="left")
    .merge(df_visits, on="Visit_ID", how="left")
    .merge(df_survey_responses, on="Attempt_ID", how="left")
)

# Ensure Visit_ID is still present
if "Visit_ID" not in survey_analytics.columns:
    survey_analytics["Visit_ID"] = np.nan  # Fill missing Visit_IDs explicitly

survey_analytics["Estimated_Cost"] = survey_analytics["Survey_Method"].map(
    {"SMS": 0.02, "Email": 0.01, "Phone": 0.04}
) * survey_analytics["Attempt_Success"].astype(int)

df_survey_analytics = survey_analytics[
    [
        "Patient_ID",
        "Age",
        "Gender",
        "Visit_ID",
        "Visit_Reason",
        "Survey_Method",
        "Satisfaction_Score",
        "Estimated_Cost",
    ]
]

# Save to CSV
df_patients.to_csv("data/patients.csv", index=False)
df_visits.to_csv("data/visits.csv", index=False)
df_survey_attempts.to_csv("data/survey_attempts.csv", index=False)
df_survey_responses.to_csv("data/survey_responses.csv", index=False)
df_survey_costs.to_csv("data/survey_costs.csv", index=False)
df_survey_analytics.to_csv("data/survey_analytics.csv", index=False)
