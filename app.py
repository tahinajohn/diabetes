import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Charger le modèle et le scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Titre de l'application
st.title('🏥 Prédiction du Diabète')
st.write('Cette application prédit le risque de diabète basé sur vos informations médicales.')

# Créer les inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Nombre de grossesses', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Niveau de glucose', min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input('Pression sanguine (mm Hg)', min_value=0, max_value=140, value=70)
    skin_thickness = st.number_input('Épaisseur de la peau (mm)', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Insuline (mu U/ml)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('IMC', min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input('Fonction du pedigree du diabète', min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input('Âge', min_value=1, max_value=120, value=30)

# Bouton de prédiction
if st.button('Prédire'):
    # Préparer les données
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, dpf, age]])
    
    # Standardiser
    input_data_scaled = scaler.transform(input_data)
    
    # Prédiction
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)
    
    # Afficher les résultats
    st.subheader('Résultats')
    
    if prediction[0] == 1:
        st.error(f'⚠️ Risque de diabète détecté (Probabilité: {probability[0][1]*100:.2f}%)')
    else:
        st.success(f'✅ Pas de risque de diabète (Probabilité: {probability[0][0]*100:.2f}%)')
    
    # Barre de progression
    st.progress(float(probability[0][1]))


