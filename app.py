import streamlit as st
import pickle

from pages.exploration import show_exploration
from pages.interpretation import show_interpretation
from pages.metrics import show_metric
from pages.model_choice import show_model_choice
from pages.prediction import show_prediction
from pages.problem import show_problem

# Configuration de la page
st.set_page_config(page_title="Prédiction Diabète", page_icon="🏥", layout="wide")

# Menu de navigation
menu = st.sidebar.radio(
    "Navigation",
    ["1. Le Problème", "2. Exploration des Données", "3. Choix du Modèle", 
     "4. Les Métriques", "5. Interprétation", "6. Prédiction"]
)

# Charger les modèles (à créer au préalable)
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('diabetes_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except:
        return None, None

# Section 1: Le Problème
if menu == "1. Le Problème":
    show_problem()
# Section 2: Exploration des Données
elif menu == "2. Exploration des Données":
    show_exploration()
# Section 3: Choix du Modèle
elif menu == "3. Choix du Modèle":
    show_model_choice()
# Section 4: Les Métriques
elif menu == "4. Les Métriques":
    show_metric()

# Section 5: Interprétation
elif menu == "5. Interprétation":
    show_interpretation()
# Section 6: Prédiction
elif menu == "6. Prédiction":
    show_prediction()