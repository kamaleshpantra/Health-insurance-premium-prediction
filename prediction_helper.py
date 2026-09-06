# codebasics ML course: codebasics.io, all rights reserverd

import pandas as pd
import joblib

# Load artifacts
model = joblib.load("artifacts/model_rest.joblib")
scaler_object = joblib.load("artifacts/scaler_rest.joblib")


def calculate_total_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease.strip(), 0) for disease in diseases)
    return total_risk_score


def calculate_income_level(income_lakhs):
    if income_lakhs < 10:
        return 1
    elif 10 <= income_lakhs <= 25:
        return 2
    elif 25 < income_lakhs <= 40:
        return 3
    else:
        return 4


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'bmi_category', 'income_level', 'income_lakhs',
        'insurance_plan', 'total_risk', 'life_style_risk', 'gender_Male',
        'region_Northwest', 'region_Southeast', 'region_Southwest',
        'marital_status_Unmarried', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried',
        'employment_status_Self-Employed'
    ]

    bmi_category_encoding = {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obesity': 4}
    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    df['age'] = input_dict.get('Age', 0)
    df['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    df['income_lakhs'] = input_dict.get('Income in Lakhs', 0)
    df['income_level'] = calculate_income_level(input_dict.get('Income in Lakhs', 0))
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict.get('Insurance Plan'), 1)
    df['bmi_category'] = bmi_category_encoding.get(input_dict.get('BMI Category'), 2)
    df['total_risk'] = calculate_total_risk(input_dict.get('Medical History', ''))
    df['life_style_risk'] = input_dict.get('Genetical Risk', 0)

    # Categorical binary encodings
    gender = input_dict.get('Gender', '')
    if gender == 'Male':
        df['gender_Male'] = 1

    region = input_dict.get('Region', '')
    if region == 'Northwest':
        df['region_Northwest'] = 1
    elif region == 'Southeast':
        df['region_Southeast'] = 1
    elif region == 'Southwest':
        df['region_Southwest'] = 1

    marital_status = input_dict.get('Marital Status', '')
    if marital_status == 'Unmarried':
        df['marital_status_Unmarried'] = 1

    smoking_status = input_dict.get('Smoking Status', '')
    if smoking_status == 'Occasional':
        df['smoking_status_Occasional'] = 1
    elif smoking_status == 'Regular':
        df['smoking_status_Regular'] = 1

    employment_status = input_dict.get('Employment Status', '')
    if employment_status == 'Salaried':
        df['employment_status_Salaried'] = 1
    elif employment_status == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1

    # Scale columns expected by scaler
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return int(prediction[0])
