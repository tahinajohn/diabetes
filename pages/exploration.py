import streamlit as st
import pandas as pd
import plotly.express as px

def show_exploration():
    st.title("📊 Exploration des Données")
    
    # Charger les données
    try:
        df = pd.read_csv('diabetes.csv')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", len(df))
        col2.metric("Diabétiques", df[df['Outcome']==1].shape[0])
        col3.metric("Non-diabétiques", df[df['Outcome']==0].shape[0])
        col4.metric("Variables", df.shape[1] - 1)
        
        tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Corrélations", "💻 Code"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='Glucose', color='Outcome', 
                                 title='Distribution du Glucose')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.box(df, x='Outcome', y='BMI', 
                           title='IMC par classe')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.imshow(df.corr(), text_auto=True, aspect="auto",
                          title='Matrice de Corrélation')
            st.plotly_chart(fig, use_container_width=True)
        
        
    
    except FileNotFoundError:
        st.warning("Fichier diabetes.csv non trouvé. Téléchargez-le depuis Kaggle.")
