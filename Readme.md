
````markdown
# 🩺 SmartHealth AI Prediction & Recommendation System

This project was developed as part of my **university learning journey in Artificial Intelligence**.  
It combines **machine learning, explainable AI, and a Streamlit interface** to predict possible diseases from user symptoms and provide simple recommendations (medications, diet, workouts, and precautions).

---

## Features

- Disease prediction using a trained **Support Vector Classifier (SVC)** model  
- **Explainable AI (SHAP)** to show which symptoms influenced predictions  
- Recommendations for:
  - Medications  
  - Precautions  
  - Diet suggestions  
  - Workout tips  
- Simple **Streamlit web app** for user interaction

---

## Project Structure

```text
SmartHealth_AI_Prediction_Recommendation_System/
│
├── app.py                       # Main Streamlit app
├── utils.py                     # Helper functions (prediction, SHAP, recommendations)
├── svc_model.pkl                 # Trained ML model
├── Datasets/                     # CSV files (symptoms, medications, diets, workouts, etc.)
├── ai_medical_assistant.ipynb    # Jupyter notebook for training & experiments
└── README.md                     # Project documentation
````

---

## ⚙️ Installation & Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/SmartHealth_AI_Prediction.git
cd SmartHealth_AI_Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the link shown in the terminal to access the web app.

---

## Example Workflow

1. Select your symptoms in the app.
2. Click **“🔍 Predict Disease”** — the model predicts the most likely disease.
3. **Explainable AI (SHAP):** a plot will be displayed showing the contribution of each symptom to the prediction:

   * Positive values → increase probability of the disease
   * Negative values → decrease probability of the disease
4. The system suggests **precautions, diets, medications, and workouts** related to that disease.

---

## Explainable AI (SHAP)

This project integrates **SHAP (SHapley Additive exPlanations)** to make predictions more transparent:

* After prediction, users can display a **SHAP plot**.
* The plot highlights which symptoms were most influential in the decision.

**Example SHAP insights:**

* Symptom A → strong positive influence (+0.45)
* Symptom B → medium influence (+0.22)
* Symptom C → negative influence (-0.18)

This makes the project not only predictive but also **interpretable**, which is crucial in healthcare AI.

---

## Learning Goals

Through this project, I practiced:

* Training and evaluating a **classification model**
* Using **Explainable AI (SHAP)** to understand model predictions
* Building a **Streamlit interface** for end-users
* Cleaning and processing messy **medical datasets**

---

## Project Screenshots

Here is a preview of the **AI Medical Assistant** app:

### 1. Enter Symptoms

Select one or more symptoms from the dropdown list.

![AI Medical Assistant Screenshot - Symptoms](Images/screenshot1.png)

---

### 2. Predict Disease

Click the **“🔍 Predict Disease”** button.
The system analyzes your symptoms and displays the predicted disease.

![AI Medical Assistant Screenshot - Predicted Disease](Images/screenshot2.png)

---

### 3. View Recommendations

Choose one of the recommendation categories (e.g., **Workout**, **Diet**, **Medications**, etc.).
The selected section expands below and displays the recommendations in a **color-highlighted format**.

![AI Medical Assistant Screenshot - Recommendations](Images/screenshot3.png)

---

### 4. Explainable AI

Scroll down to the **Explainable AI** section to see:

* A SHAP visualization of how symptoms contributed to the prediction
* A textual explanation of the model’s reasoning

![AI Medical Assistant Screenshot - Explainable AI](Images/screenshot4.png)

```

✅ This version:

- Uses proper Markdown spacing and line breaks.  
- All headings and sections render nicely on GitHub.  
- Code blocks, file tree, and command examples are formatted clearly.  
- Screenshots and step descriptions are easy to follow.  

If you want, I can also **add a concise one-line repository description** that fits GitHub’s description field perfectly. Do you want me to do that?
```
