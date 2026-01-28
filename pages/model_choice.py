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
    
    with st.expander("💻 Voir le code d'entraînement"):
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Préparation des données
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Sauvegarde
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
        """, language='python')
