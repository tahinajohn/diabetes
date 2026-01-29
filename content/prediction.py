import streamlit as st
import numpy as np
import pickle

import pandas as pd


@st.cache_resource
def load_model(filepath='model/diabetes_model.pkl'):
    """Charge le modèle et les artefacts sauvegardés."""
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    return model_artifacts


def create_features(X):
    """Crée des features supplémentaires (doit correspondre à l'entraînement)."""
    X_new = X.copy()
    
    # Interactions cliniquement pertinentes
    X_new['BMI_Age'] = X_new['BMI'] * X_new['Age']
    X_new['Glucose_Insulin'] = X_new['Glucose'] * X_new['Insulin']
    X_new['Glucose_BMI'] = X_new['Glucose'] * X_new['BMI']
    
    # Indicateurs de risque
    X_new['High_Risk'] = ((X_new['Age'] > 50) & (X_new['BMI'] > 30)).astype(int)
    X_new['Pregnancy_Risk'] = ((X_new['Pregnancies'] > 6) & (X_new['Age'] > 30)).astype(int)
    
    return X_new


def predict_diabetes(patient_data, model_artifacts):
    """
    Prédit le risque de diabète pour un nouveau patient.
    
    Parameters:
    -----------
    patient_data : dict
        Dictionnaire contenant les données du patient avec les clés:
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    
    model_artifacts : dict
        Dictionnaire contenant le modèle et les transformateurs
    
    Returns:
    --------
    dict : Résultats de la prédiction
    """
    # Créer un DataFrame avec les données du patient
    patient_df = pd.DataFrame([patient_data])
    
    # Remplacer les 0 par NaN pour les colonnes concernées
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        if col in patient_df.columns and patient_df[col].iloc[0] == 0:
            patient_df[col] = np.nan
    
    # Imputation
    patient_imputed = pd.DataFrame(
        model_artifacts['imputer'].transform(patient_df),
        columns=patient_df.columns
    )
    
    # Feature engineering
    patient_engineered = create_features(patient_imputed)
    
    # Standardisation
    patient_scaled = pd.DataFrame(
        model_artifacts['scaler'].transform(patient_engineered),
        columns=patient_engineered.columns
    )
    
    # Prédiction
    probability = model_artifacts['model'].predict_proba(patient_scaled)[0, 1]
    prediction = int(probability >= model_artifacts['optimal_threshold'])
    
    # Niveau de risque
    if probability < 0.3:
        risk_level = "Faible"
        risk_color = "🟢"
    elif probability < 0.6:
        risk_level = "Modéré"
        risk_color = "🟡"
    else:
        risk_level = "Élevé"
        risk_color = "🔴"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'risk_level': risk_level,
        'risk_color': risk_color,
        'threshold_used': model_artifacts['optimal_threshold']
    }


def predict_single_patient():
    """Interface pour prédire un seul patient."""
    print("="*70)
    print("🏥 PRÉDICTION DU DIABÈTE - PATIENT UNIQUE")
    print("="*70)
    
    # Charger le modèle
    print("\n📥 Chargement du modèle...")
    model_artifacts = load_model()
    print("   ✅ Modèle chargé avec succès!")
    
    # Exemple de patient
    print("\n👤 Données du patient exemple:")
    patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,  # Valeur manquante
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    for key, value in patient.items():
        print(f"   • {key:30s}: {value}")
    
    # Prédiction
    print("\n🔮 Prédiction en cours...")
    result = predict_diabetes(patient, model_artifacts)
    
    # Affichage des résultats
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE LA PRÉDICTION")
    print("="*70)
    print(f"\n{result['risk_color']} Prédiction:           {result['prediction']}")
    print(f"   Probabilité:          {result['probability']:.2%}")
    print(f"   Niveau de risque:     {result['risk_level']}")
    print(f"   Seuil utilisé:        {result['threshold_used']:.2f}")
    print("\n" + "="*70)
    
    return result



def show_prediction():
    st.title("🔮 Prédiction Interactive")
    
    model = load_model()
    
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
        
        if st.button("🔍 Prédire le Risque", width='stretch'):
            # input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
            #                        insulin, bmi, dpf, age]])
            
            input_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            result = predict_diabetes(input_data, model)
            probability = result["probability"]
            
            if result["prediction"] == 1:
                st.error(f"⚠️ **Risque de diabète détecté**")
                st.metric("Probabilité d'être diabétique", f"{probability*100:.1f}%")
            else:
                st.success(f"✅ **Pas de risque détecté**")
                st.metric("Probabilité d'être diabétique", f"{probability*100:.1f}%")
            
            st.progress(float(probability))
            st.caption("⚠️ Cette prédiction est à titre informatif uniquement.")