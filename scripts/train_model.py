"""
train_model.py
==============
Script d'entraînement du modèle Random Forest pour la prédiction du diabète.
Ce fichier gère le chargement des données, le preprocessing et l'entraînement du modèle.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score,
    f1_score
)
# from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_data():
    """Charge le dataset Pima Indians Diabetes."""
    
    df = pd.read_csv("data/diabetes.csv")
    
    print(f"✅ Données chargées: {df.shape}")
    print(f"   - Diabétiques: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
    print(f"   - Non-diabétiques: {(df['Outcome']==0).sum()} ({(1-df['Outcome'].mean())*100:.1f}%)")
    
    return df


def preprocess_data(df):
    """
    Préprocessing des données:
    - Remplacement des 0 par NaN
    - Imputation avec la médiane
    - Traitement des outliers
    - Feature engineering
    """
    print("\n🔧 Preprocessing des données...")
    
    # Copie pour éviter de modifier l'original
    df_processed = df.copy()
    
    # 1. Remplacer les 0 par NaN pour les colonnes concernées
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df_processed[col] = df_processed[col].replace(0, np.nan)
    
    print(f"   ✓ Valeurs 0 remplacées par NaN pour {len(zero_cols)} colonnes")
    
    # 2. Séparation features/cible
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # 3. Split train/test (stratifié)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   ✓ Split train/test: {X_train.shape[0]} / {X_test.shape[0]}")
    
    # 4. Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    print("   ✓ Imputation des valeurs manquantes (médiane)")
    
    # 5. Traitement des outliers (capping)
    X_train_capped = cap_outliers(X_train_imputed)
    X_test_capped = X_test_imputed.copy()
    for col in X_test_capped.columns:
        lower_bound = X_train_imputed[col].quantile(0.01)
        upper_bound = X_train_imputed[col].quantile(0.99)
        X_test_capped[col] = X_test_capped[col].clip(lower=lower_bound, upper=upper_bound)
    print("   ✓ Outliers traités (capping aux percentiles 1 et 99)")
    
    # 6. Feature engineering
    X_train_engineered = create_features(X_train_capped)
    X_test_engineered = create_features(X_test_capped)
    print(f"   ✓ Feature engineering: {X_train_engineered.shape[1]} features")
    
    # 7. Standardisation
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_engineered),
        columns=X_train_engineered.columns,
        index=X_train_engineered.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_engineered),
        columns=X_test_engineered.columns,
        index=X_test_engineered.index
    )
    print("   ✓ Standardisation appliquée")
    
    # 8. SMOTE pour équilibrer les classes
    #smote = SMOTE(random_state=RANDOM_STATE)
    #X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    #print(f"   ✓ SMOTE appliqué: {len(y_train)} → {len(y_train_balanced)} échantillons")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, imputer, scaler


def cap_outliers(df, lower_percentile=1, upper_percentile=99):
    """Traite les outliers en les cappant aux percentiles spécifiés."""
    df_capped = df.copy()
    for col in df.columns:
        lower_bound = df[col].quantile(lower_percentile/100)
        upper_bound = df[col].quantile(upper_percentile/100)
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    return df_capped


def create_features(X):
    """Crée des features supplémentaires par engineering."""
    X_new = X.copy()
    
    # Interactions cliniquement pertinentes
    X_new['BMI_Age'] = X_new['BMI'] * X_new['Age']
    X_new['Glucose_Insulin'] = X_new['Glucose'] * X_new['Insulin']
    X_new['Glucose_BMI'] = X_new['Glucose'] * X_new['BMI']
    
    # Indicateurs de risque
    X_new['High_Risk'] = ((X_new['Age'] > 50) & (X_new['BMI'] > 30)).astype(int)
    X_new['Pregnancy_Risk'] = ((X_new['Pregnancies'] > 6) & (X_new['Age'] > 30)).astype(int)
    
    return X_new


def train_random_forest(X_train, y_train):
    """
    Entraîne un modèle Random Forest avec GridSearch pour l'optimisation.
    """
    print("\n🌲 Entraînement du Random Forest...")
    
    # Modèle de base
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # Grille de paramètres
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Stratégie de validation croisée
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Grid Search
    print("   🔍 Grid Search en cours...")
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"   ✅ Meilleurs paramètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"      - {param}: {value}")
    print(f"   ✅ Meilleur score AUC (CV): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle sur le test set."""
    print("\n📊 Évaluation du modèle...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   • Accuracy:  {accuracy:.4f}")
    print(f"   • ROC-AUC:   {auc:.4f}")
    print(f"   • F1-Score:  {f1:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non Diabétique', 'Diabétique']))
    
    print("📊 Matrice de Confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                     Prédiction")
    print(f"                Non-Diab  Diabétique")
    print(f"Réel Non-Diab      {cm[0,0]:3d}        {cm[0,1]:3d}")
    print(f"     Diabétique    {cm[1,0]:3d}        {cm[1,1]:3d}")
    
    # Calculs supplémentaires
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"\n   • Sensitivity (Recall): {sensitivity:.4f}")
    print(f"   • Specificity:          {specificity:.4f}")
    
    res = {
        "y_pred_proba": y_pred_proba,
        "accuracy": accuracy, 
        "auc": auc,
        "f1-score": f1,
        "cm":cm
    }
    return res


def find_optimal_threshold(y_test, y_pred_proba):
    """Trouve le seuil optimal pour maximiser le F1-score."""
    print("\n🎯 Recherche du seuil optimal...")
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   ✅ Seuil optimal: {best_threshold:.2f}")
    print(f"   ✅ F1-Score à ce seuil: {best_f1:.4f}")
    
    return best_threshold


def analyze_feature_importance(model, feature_names):
    """Analyse l'importance des features."""
    print("\n🔍 Feature Importance:")
    
    # Obtenir les importances
    importances = model.feature_importances_
    
    # Créer un DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n   Top 10 features les plus importantes:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"      {row['Feature']:30s} : {row['Importance']:.4f}")
    
    return importance_df


def save_model(model, imputer, scaler, feature_names, optimal_threshold, evaluation,filepath='model/diabetes_model.pkl'):
    """Sauvegarde le modèle et tous les artefacts nécessaires."""
    print(f"\n💾 Sauvegarde du modèle dans '{filepath}'...")
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': feature_names,
        'optimal_threshold': optimal_threshold,
        'evaluation': evaluation
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print("   ✅ Modèle sauvegardé avec succès!")


def main():
    """Fonction principale."""
    print("="*70)
    print("🏥 PRÉDICTION DU DIABÈTE - RANDOM FOREST")
    print("="*70)
    
    # 1. Chargement des données
    df = load_data()
    
    # 2. Preprocessing
    X_train, X_test, y_train, y_test, imputer, scaler = preprocess_data(df)
    
    # 3. Entraînement
    model = train_random_forest(X_train, y_train)
    
    # 4. Évaluation
    evaluation = evaluate_model(model, X_test, y_test)
    
    # 5. Seuil optimal
    optimal_threshold = find_optimal_threshold(y_test, evaluation["y_pred_proba"])
    
    # 6. Feature importance
    importance_df = analyze_feature_importance(model, X_train.columns.tolist())
    
    # 7. Sauvegarde
    save_model(model, imputer, scaler, X_train.columns.tolist(), optimal_threshold, evaluation)
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    print("   Le modèle est sauvegardé dans 'diabetes_model.pkl'")


if __name__ == "__main__":
    main()