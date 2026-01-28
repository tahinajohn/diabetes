import streamlit as st
import numpy as np
from app import load_model 

def show_prediction():
    st.title("🔮 Prédiction Interactive")
    
    model, scaler = load_model()
    
    if model is None:
        st.warning("Modèle non chargé. Créez d'abord votre modèle avec la section 3.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Nombre de grossesses", 0, 20, 0)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 120)
            blood_pressure = st.slider("Pression sanguine (mm Hg)", 0, 140, 70)
            skin_thickness = st.slider("Épaisseur peau (mm)", 0, 100, 20)
        
        with col2:
            insulin = st.slider("Insuline (mu U/ml)", 0, 900, 80)
            bmi = st.slider("IMC", 0.0, 70.0, 25.0)
            dpf = st.slider("Fonction pedigree", 0.0, 3.0, 0.5)
            age = st.number_input("Âge", 1, 120, 30)
        
        if st.button("🔍 Prédire le Risque", use_container_width=True):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                   insulin, bmi, dpf, age]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            if prediction[0] == 1:
                st.error(f"⚠️ **Risque de diabète détecté**")
                st.metric("Probabilité", f"{probability[0][1]*100:.1f}%")
            else:
                st.success(f"✅ **Pas de risque détecté**")
                st.metric("Probabilité (sain)", f"{probability[0][0]*100:.1f}%")
            
            st.progress(float(probability[0][1]))
            st.caption("⚠️ Cette prédiction est à titre informatif uniquement.")