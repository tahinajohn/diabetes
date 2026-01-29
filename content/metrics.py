import streamlit as st
import plotly.express as px
import numpy as np
import pickle

@st.cache_resource
def load_model(filepath='model/diabetes_model.pkl'):
    """Charge le modèle et les artefacts sauvegardés."""
    with open(filepath, 'rb') as f:
        model_artifacts = pickle.load(f)
    return model_artifacts

def show_metric():
    st.title("📊 Les Métriques")
    model = load_model()
    acc = model["evaluation"]["accuracy"]
    f1 = model["evaluation"]["f1-score"]
    auc = model["evaluation"]["auc"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("f1-score", f"{f1:.2f}")
    col3.metric("AUC", f"{auc:.2f}")
    
    st.markdown("### Matrice de Confusion")
    confusion = np.array(model["evaluation"]["cm"])
    fig = px.imshow(confusion, text_auto=True,
                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                    x=['Négatif', 'Positif'],
                    y=['Négatif', 'Positif'],
                    color_continuous_scale='Blues')
    st.plotly_chart(fig, width='stretch')
    
