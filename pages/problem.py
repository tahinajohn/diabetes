import streamlit as st
import pandas as pd

def show_problem():
    st.title("🏥 Le Problème")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Contexte
        Le diabète est une maladie chronique qui affecte des millions de personnes dans le monde. 
        La détection précoce est cruciale pour prévenir les complications graves.
        
        ### Objectif du Projet
        Développer un modèle de machine learning capable de prédire si une personne est susceptible 
        d'avoir le diabète en se basant sur des indicateurs médicaux.
        """)
    
    with col2:
        st.info("**Dataset**: Pima Indians Diabetes")
        st.metric("Observations", "768")
        st.metric("Features", "8")
    
    st.markdown("### Variables du Dataset")
    features = {
        "Variable": ["Grossesses", "Glucose", "Pression", "Épaisseur peau", 
                    "Insuline", "IMC", "Fonction pedigree", "Âge"],
        "Description": [
            "Nombre de fois enceinte",
            "Concentration glucose plasmatique",
            "Pression artérielle diastolique (mm Hg)",
            "Épaisseur du pli cutané (mm)",
            "Insuline sérique (mu U/ml)",
            "Indice de masse corporelle",
            "Fonction du pedigree du diabète",
            "Âge en années"
        ]
    }
    st.table(pd.DataFrame(features))
