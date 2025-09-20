# utils.py
import joblib
import numpy as np
import shap
import pandas as pd
import streamlit.components.v1 as components
import ast


# ---------------------------
# 1. Load Trained Model & Data
# ---------------------------

# Load the trained Support Vector Classifier model
svc_model = joblib.load(open("svc_model.pkl", "rb"))

# Symptoms dictionary mapping symptom name to index
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43,
    'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49,
    'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53,
    'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57,
    'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64,
    'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68,
    'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
    'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
    'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
    'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109,
    'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
    'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Diseases list mapping prediction index to disease name
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine',
    7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice',
    29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E',
    3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins',
    26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}


# ---------------------------
# 2. Prediction Function
# ---------------------------

def predict_disease(user_symptoms):
    """
    Predicts the disease based on user-selected symptoms.
    
    Args:
        user_symptoms (list[str]): List of symptom names.
    
    Returns:
        tuple: (predicted_disease_name, input_vector)
    """
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in user_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    prediction = diseases_list[svc_model.predict([input_vector])[0]]
    return prediction, input_vector


# ---------------------------
# 3. Explainable AI Functions
# ---------------------------

def st_shap(plot, height=None):
    """
    Display SHAP plots inside Streamlit using HTML.
    """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 300)


def explain_with_shap(user_vector):
    """
    Explain the model’s prediction using SHAP.
    
    Args:
        user_vector (list[float]): Encoded symptom input vector.
    
    Returns:
        tuple: (shap_plot, explanation_text)
    """
    all_symptoms = list(symptoms_dict.keys())

    # Create SHAP explainer
    explainer = shap.Explainer(
        svc_model,
        np.zeros((1, len(all_symptoms))),
        feature_names=all_symptoms
    )

    # Get SHAP values for user input
    shap_values = explainer(np.array(user_vector).reshape(1, -1))

    # Flatten contributions
    shap_contributions = [
        (symptom, float(impact[0]) if hasattr(impact, "__len__") else float(impact))
        for symptom, impact in zip(all_symptoms, shap_values.values[0])
    ]

    # Sort by importance
    shap_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # Pick top 5 factors
    top_symptoms = [
        f"{symptom} (impact: {impact:.4f})"
        for symptom, impact in shap_contributions[:5]
    ]
    explanation_text = "📊 Top factors influencing the prediction:\n" + "\n".join(top_symptoms)

    # Make SHAP plot
    shap.initjs()
    shap_plot = shap.force_plot(
        explainer.expected_value[0],
        shap_values.values[0],
        all_symptoms
    )

    return shap_plot, explanation_text


# ---------------------------
# 4. Get Disease Recommendations
# ---------------------------

# Load datasets
symptoms = pd.read_csv('Datasets/symptoms_df.csv')
precautions = pd.read_csv('Datasets/precautions_df.csv')
workout = pd.read_csv('Datasets/workout_df.csv')
description = pd.read_csv('Datasets/description.csv')
medications = pd.read_csv('Datasets/medications.csv')
diets = pd.read_csv('Datasets/diets.csv')


def _clean_and_flatten(raw_values):
    """
    Helper function: Cleans and flattens messy dataset values into a list of strings.
    Handles:
    - Lists, tuples, sets
    - Stringified lists
    - Comma-separated strings
    - NaN / None
    - Other datatypes (converted to string)
    """
    cleaned = []

    for val in raw_values:
        if pd.isna(val):
            continue

        if isinstance(val, (list, tuple, set)):
            for it in val:
                if not pd.isna(it):
                    s = str(it).strip()
                    if s:
                        cleaned.append(s)
            continue

        if isinstance(val, str):
            s = val.strip()
            if not s:
                continue

            # Try literal_eval
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None

            if isinstance(parsed, (list, tuple, set)):
                for it in parsed:
                    if not pd.isna(it):
                        item = str(it).strip()
                        if item:
                            cleaned.append(item)
                continue

            if ',' in s:
                parts = [p.strip().strip("'\"") for p in s.strip("[]").split(',') if p.strip()]
                cleaned.extend(parts)
                continue

            cleaned_val = s.strip("[]'\" ")
            if cleaned_val:
                cleaned.append(cleaned_val)
            continue

        cleaned.append(str(val).strip())

    return [c for c in cleaned if c]


def get_recommendations(disease):
    """
    Get description, precautions, medications, diets, and workout recommendations for a disease.
    
    Args:
        disease (str): Predicted disease name.
    
    Returns:
        tuple: (desc, precaution, medication, diet, workout)
    """
    # 1. Description
    desc = " ".join(
        description[description['Disease'] == disease]['Description'].astype(str).values
    )

    # 2. Precautions
    prec_raw = precautions[precautions['Disease'] == disease][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ].values.flatten().tolist()
    precaution = [str(p).strip() for p in prec_raw if (not pd.isna(p)) and str(p).strip()]

    # 3. Medications
    med_raw = medications[medications['Disease'] == disease]['Medication'].astype(object).values.tolist()
    medication = _clean_and_flatten(med_raw)

    # 4. Diets
    diet_raw = diets[diets['Disease'] == disease]['Diet'].astype(object).values.tolist()
    diet = _clean_and_flatten(diet_raw)

    # 5. Workouts
    wrk_raw = workout[workout['disease'] == disease]['workout'].astype(object).values.tolist()
    wrkout = _clean_and_flatten(wrk_raw)

    return desc, precaution, medication, diet, wrkout
