# ---------------------------
# AI Medical Assistant - Streamlit Web App
# ---------------------------
# Predicts diseases based on selected symptoms,
# provides recommendations, and includes Explainable AI.
# ---------------------------

import streamlit as st
import ast
from utils import (
    predict_disease,
    st_shap,
    explain_with_shap,
    get_recommendations,
    symptoms_dict  
)

# ---------------------------
# 1. Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="AI Medical Assistant", page_icon="💊", layout="centered")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #1ABC9C; font-size:50px;'>🩺 AI Medical Assistant</h1>
    """,
    unsafe_allow_html=True
)
st.write("---")


# ---------------------------
# 2. Symptom Selection
# ---------------------------
st.markdown(
    "<h2 style='font-size:28px; text-align:left;'>Select your symptoms:</h2>",
    unsafe_allow_html=True
)

selected_symptoms = st.multiselect(
    label="",
    options=list(symptoms_dict.keys())
)
st.write("---")


# ---------------------------
# 3. Predict Disease Button
# ---------------------------
if st.button("🔍 Predict Disease", key="predict_btn"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom!")
    else:
        predicted_disease, user_vector = predict_disease(selected_symptoms)
        st.session_state.predicted_disease = predicted_disease
        st.session_state.user_vector = user_vector


# ---------------------------
# 4. Show Predicted Disease
# ---------------------------
if "predicted_disease" in st.session_state:
    st.markdown(f"""
    <div style='
        width:400px; margin:20px auto; padding:25px;
        background-color:#FFFFFF; border-radius:15px;
        border:3px solid #1ABC9C; font-size:28px;
        font-weight:bold; text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.1);
    '>
        🩺 <span style='color:#1ABC9C;'>Predicted Disease:</span><br> {st.session_state.predicted_disease}
    </div>
    """, unsafe_allow_html=True)


# ---------------------------
# 5. Recommendation Buttons
# ---------------------------
button_styles = {
    "Description": {"color": "#8FD694", "icon": "📖"},
    "Precautions": {"color": "#F9D67A", "icon": "🛡️"},
    "Medications": {"color": "#FFABAB", "icon": "💊"},
    "Diets": {"color": "#A0C4FF", "icon": "🥗"},
    "Workout": {"color": "#FFD6A5", "icon": "🏋️"}
}

cols = st.columns(5)


def section_header_inline(icon, title):
    """Helper → render section header inline with emoji + title"""
    st.markdown(f"<h3 style='text-align:left;'>{icon} {title}</h3>", unsafe_allow_html=True)


for i, btn_name in enumerate(button_styles.keys()):
    col = cols[i]
    style = button_styles[btn_name]

    if "predicted_disease" in st.session_state:
        disease = st.session_state.predicted_disease

        if col.button(f"{style['icon']} {btn_name}", key=f"{btn_name}_btn", use_container_width=True):
            desc, precaution, medication, diet, wrkout = get_recommendations(disease)

            section_header_inline(style['icon'], btn_name)

            if btn_name == "Description":
                st.info(desc if desc.strip() else "No description available.")

            elif btn_name == "Precautions":
                if precaution:
                    for idx, p in enumerate(precaution, start=1):
                        st.success(f"{idx}. {p}")
                else:
                    st.info("No precautions available.")

            elif btn_name == "Medications":
                if medication:
                    for idx, m in enumerate(medication, start=1):
                        if isinstance(m, str) and m.strip().startswith("["):
                            try:
                                parsed = ast.literal_eval(m)
                                if isinstance(parsed, (list, tuple)):
                                    for j, sub in enumerate(parsed, start=1):
                                        st.warning(f"{idx}.{j}. {str(sub).strip()}")
                                    continue
                            except Exception:
                                pass
                        st.warning(f"{idx}. {str(m).strip()}")
                else:
                    st.info("No medications available.")

            elif btn_name == "Diets":
                if diet:
                    for idx, d in enumerate(diet, start=1):
                        if isinstance(d, str) and d.strip().startswith("["):
                            try:
                                parsed = ast.literal_eval(d)
                                if isinstance(parsed, (list, tuple)):
                                    for j, sub in enumerate(parsed, start=1):
                                        st.success(f"{idx}.{j}. {str(sub).strip()}")
                                    continue
                            except Exception:
                                pass
                        st.success(f"{idx}. {str(d).strip()}")
                else:
                    st.info("No diets available.")

            elif btn_name == "Workout":
                if wrkout:
                    for idx, w in enumerate(wrkout, start=1):
                        st.info(f"{idx}. {str(w).strip()}")
                else:
                    st.info("No workout recommendations available.")
    else:
        col.markdown(
            f"<button style='background-color:{style['color']}; color:black; "
            f"font-weight:bold; width:100%; height:45px; border-radius:10px;' disabled>"
            f"{style['icon']} {btn_name}</button>",
            unsafe_allow_html=True
        )


# ---------------------------
# 6. Background Color
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #F4F9F9; }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------
# 7. Explainable AI Section
# ---------------------------
if "predicted_disease" in st.session_state and "user_vector" in st.session_state:
    st.write("---")
    st.markdown(
        "<h2 style='font-size:26px; text-align:left; color:#1ABC9C;'>🤖 Explainable AI</h2>",
        unsafe_allow_html=True
    )

    shap_plot, explanation_text = explain_with_shap(st.session_state.user_vector)

    st.info(explanation_text)
    st_shap(shap_plot)
