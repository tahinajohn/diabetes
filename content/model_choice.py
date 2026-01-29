import streamlit as st

def show_model_choice():
    st.title("🧠 Choix du Modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Logistic Regression**")
        st.write("✓ Simple")
        st.write("✓ Interprétable")
        st.write("✗ Performance limitée")
    
    with col2:
        st.success("**Random Forest** ⭐")
        st.write("✓ Meilleure performance")
        st.write("✓ Gère non-linéarité")
        st.write("✓ Feature importance")
    
    with col3:
        st.info("**XGBoost**")
        st.write("✓ Très performant")
        st.write("✗ Plus complexe")
        st.write("✗ Temps d'entraînement")
    
    st.markdown("### Pourquoi Random Forest ?")
    st.markdown("""
    - Gère bien les données non-linéaires
    - Robuste aux valeurs aberrantes
    - Importance des features intégrée
    - Pas de surapprentissage (avec bons paramètres)
    - Performance équilibrée
    """)
    
   