import streamlit as st
import plotly.graph_objects as go

def show_interpretation():
    st.title("💡 Interprétation")
    
    st.markdown("### Importance des Features")
    features = ['Glucose', 'IMC', 'Âge', 'Pedigree', 'Pression', 'Insuline', 'Grossesses', 'Peau']
    importance = [25, 18, 16, 14, 10, 8, 5, 4]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color='teal'
    ))
    fig.update_layout(title='Importance des Variables', xaxis_title='Importance (%)')
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Points Forts**")
        st.markdown("""
        - Modèle interprétable
        - Glucose = meilleur prédicteur
        - Stable et reproductible
        """)
    
    with col2:
        st.error("**Limitations**")
        st.markdown("""
        - 25 faux négatifs (cas manqués)
        - Dataset limité à une population
        - Valeurs manquantes codées en 0
        - Déséquilibre des classes
        - Ne remplace pas un diagnostic médical
        """)
    
    st.warning("⚠️ **Recommandations**: Utiliser comme outil de dépistage préliminaire uniquement. Toujours consulter un professionnel de santé.")
