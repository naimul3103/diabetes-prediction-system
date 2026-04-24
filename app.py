import pickle
import numpy as np
import pandas as pd
import gradio as gr

#  STEP 1 — Load model from best_model.pkl using pickle
with open('best_model.pkl', 'rb') as f:
    payload = pickle.load(f)

model = payload['model']           
feature_cols = payload['feature_cols']    
class_names = payload['class_names']    
test_acc = payload['test_accuracy']
test_auc = payload['test_roc_auc']

print(f'Type : {type(model).__name__}')
print(f'Test Accuracy : {test_acc}  |  Test ROC-AUC : {test_auc}')
print(f'Features ({len(feature_cols)}): {feature_cols}')


#  STEP 2 — Prediction function
def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                     insulin, bmi, diabetes_pedigree, age):

    # Engineered features 
    if bmi < 18.5: bmi_cat = 0
    elif bmi < 25.0: bmi_cat = 1
    elif bmi < 30.0: bmi_cat = 2
    else: bmi_cat = 3

    gi_ratio = glucose / (insulin + 1)

    if age < 30: age_grp = 0
    elif age < 45: age_grp = 1
    elif age < 60: age_grp = 2
    else: age_grp = 3

    # Build DataFrame 
    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree, age,
        bmi_cat, gi_ratio, age_grp
    ]], columns=feature_cols)

    # Predict using pickle-loaded model
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    prob_neg = probability[0] * 100
    prob_pos = probability[1] * 100

    if prediction == 1:
        verdict = 'DIABETIC'
        risk_level = f'HIGH RISK  ({prob_pos:.1f}% confidence)'
        advice = ('Please consult an endocrinologist or general physician as soon '
                      'as possible. Early intervention significantly reduces complications.')
    else:
        verdict = 'NON-DIABETIC'
        risk_level = f'LOW RISK  ({prob_neg:.1f}% confidence)'
        advice= ('No diabetes detected. Maintain a balanced diet, regular exercise, ''and periodic health check-ups.')

    result = (
        f'RESULT: {verdict}\n'
        f'RISK LEVEL : {risk_level}\n'
        f'\n'
        f'Probability Breakdown:\n'
        f'Non-Diabetic : {prob_neg:.1f}%\n'
        f'Diabetic : {prob_pos:.1f}%\n'
        f'\n'
        f'Engineered Features Computed:\n'
        f'BMI Category : {bmi_cat}  (0=Underweight 1=Normal 2=Overweight 3=Obese)\n'
        f'Glucose/Insulin Ratio : {gi_ratio:.3f}\n'
        f'Age Group : {age_grp}  (0=<30 1=30-45 2=45-60 3=60+)\n'
        f'\n'
        f'Recommendation:\n'
        f'{advice}\n'
        f'\n'
        f'[Source: best_model.pkl | Tuned Random Forest | '
        f'Test Acc: {test_acc} | Test AUC: {test_auc}]'
    )
    return result


#  STEP 3 — Build Gradio Interface
with gr.Blocks(title='Diabetes Prediction System') as demo:

    gr.Markdown(f"""
    # <center>Diabetes Prediction System </center>
    <center>Powered by a Tuned Random Forest Classifier</center>
    Enter the patient's clinical details and click **Predict**.  
    """)

    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=1):
            gr.Markdown('#### Patient Clinical Details')
            pregnancies = gr.Number(
                label='Pregnancies',
                value=1, minimum=0, maximum=20,
                info='Number of times pregnant'
            )
            glucose = gr.Slider(
                label='Glucose (mg/dL)',
                minimum=44, maximum=200, value=110, step=1,
                info='Plasma glucose — 2-hr oral glucose tolerance test'
            )
            blood_pressure = gr.Slider(
                label='Blood Pressure (mmHg)',
                minimum=24, maximum=122, value=70, step=1,
                info='Diastolic blood pressure'
            )
            skin_thickness = gr.Slider(
                label='Skin Thickness (mm)',
                minimum=7, maximum=63, value=23, step=1,
                info='Triceps skin fold thickness'
            )
            insulin = gr.Slider(
                label='Insulin (uU/mL)',
                minimum=14, maximum=846, value=80, step=1,
                info='2-Hour serum insulin level'
            )
            bmi = gr.Slider(
                label='BMI',
                minimum=18.0, maximum=67.1, value=28.0, step=0.1,
                info='Body Mass Index (weight kg / height m^2)'
            )
            dpf = gr.Slider(
                label='Diabetes Pedigree Function',
                minimum=0.078, maximum=2.42, value=0.47, step=0.001,
                info='Genetic risk score based on family history'
            )
            age = gr.Slider(
                label='Age (years)',
                minimum=21, maximum=81, value=33, step=1,
                info='Age of the patient'
            )
            predict_btn = gr.Button('Predict Diabetes Risk', variant='primary', size='lg')

        # Right column — output
        with gr.Column(scale=1):
            gr.Markdown('#### Prediction Result')
            
            output = gr.Textbox(
                label='Diagnosis Output',
                lines=22,
                max_lines=25
            )

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age],
        outputs=output
    )

    gr.Markdown('Quick Test Examples (click to auto-fill)')
    gr.Examples(
        examples=[
            [6, 148, 72, 35,   0, 33.6, 0.627, 50],
            [1,  89, 66, 23,  94, 28.1, 0.167, 21],
            [8, 183, 64,  0,   0, 23.3, 0.672, 32],
            [2,  99, 60, 17, 160, 36.6, 0.453, 21],
        ],
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, dpf, age],
        label='High Risk / Low Risk / High Risk / Low Risk'
    )

    gr.Markdown("""**Disclaimer:** This tool is for educational purposes only.  
    Always consult a qualified healthcare professional for medical advice.""")
demo.launch(share=True, debug=True, theme=gr.themes.Soft())