import streamlit as st
import plotly.express as px
import numpy as np

def show_metric():
    st.title("📊 Les Métriques")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "78%", "↑ 5%")
    col2.metric("Precision", "74%", "↑ 3%")
    col3.metric("Recall", "71%", "↑ 2%")
    col4.metric("AUC-ROC", "0.82", "↑ 0.05")
    
    st.markdown("### Matrice de Confusion")
    confusion = np.array([[85, 15], [19, 35]])
    fig = px.imshow(confusion, text_auto=True,
                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                    x=['Négatif', 'Positif'],
                    y=['Négatif', 'Positif'],
                    color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
