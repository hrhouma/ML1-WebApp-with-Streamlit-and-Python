# Importations nécessaires pour l'application web et le traitement de données
import streamlit as st  # Framework pour créer des applications web interactives en Python
import pandas as pd  # Bibliothèque pour la manipulation et l'analyse des données
from sklearn.preprocessing import LabelEncoder  # Outil pour encoder les étiquettes textuelles en valeurs numériques
from sklearn.model_selection import train_test_split  # Fonction pour diviser les données en ensembles de test et d'entraînement
from sklearn.svm import SVC  # Importe le modèle de Machine à Vecteurs de Support pour la classification
from sklearn.metrics import precision_score, recall_score  # Métriques pour évaluer la précision et le rappel du modèle
# Importations pour afficher diverses métriques de performance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression  # Importe le modèle de régression logistique
from sklearn.ensemble import RandomForestClassifier  # Importe le modèle de forêt aléatoire

# Définition de la fonction principale de l'application
def main():
    # Affichage du titre de l'application web
    st.title("Binary Classification WebApp")    
    # Description courte sous le titre
    st.markdown("Are your mushroom edible or poisonous? 🍄")

    # Configuration du titre et de la description dans la barre latérale
    st.sidebar.title("Binary Classification")
    st.sidebar.markdown("Are your mushroom edible or poisonous?")

    # Décoration de la fonction load_data avec st.cache pour la mise en cache des données chargées
    @st.cache_data(persist=True)
    def load_data():
        # Lecture des données à partir d'un fichier CSV
        data = pd.read_csv('mushrooms.csv')
        # Création d'un encodeur pour transformer les données textuelles en numériques
        label = LabelEncoder()
        # Encodage de chaque colonne dans le DataFrame
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data  # Retourne le DataFrame encodé

    # Fonction pour diviser les données en ensembles d'entraînement et de test
    @st.cache_data(persist=True)
    def split(df):
        # Séparation de la colonne cible 'type'
        y = df.type
        # Suppression de la colonne cible du DataFrame pour obtenir les caractéristiques
        x = df.drop(columns=['type'])
        # Division des données en ensembles d'entraînement et de test (70% train, 30% test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test  # Retourne les ensembles divisés

    # Fonction pour afficher les métriques sélectionnées
    def plot_metrics(metrics_list):
        # Affiche la matrice de confusion si sélectionnée
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()  # Création d'une figure pour le graphique
            # Affichage de la matrice de confusion à partir de l'estimateur
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names, ax=ax)
            st.pyplot(fig)  # Affiche le graphique sur l'application web

        # Affiche la courbe ROC si sélectionnée
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()  # Création d'une figure pour le graphique
            # Affichage de la courbe ROC à partir de l'estimateur
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)  # Affiche le graphique sur l'application web

        # Affiche la courbe Précision-Rappel si sélectionnée
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()  # Création d'une figure pour le graphique
            # Affichage de la courbe Précision-Rappel à partir de l'estimateur
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)  # Affiche le graphique sur l'application web

    # Chargement des données et préparation des ensembles d'entraînement et de test
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']  # Noms des classes pour l'affichage

    # Configuration de la sélection du classificateur dans la barre latérale
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    
    # Configuration pour le modèle SVM
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        # Paramètres ajustables pour le modèle SVM dans la barre latérale
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'auto')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        # Bouton pour lancer la classification
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            model.fit(x_train, y_train)  # Entraînement du modèle sur les données
            accuracy = model.score(x_test, y_test)  # Calcul de la précision du modèle
            y_pred = model.predict(x_test)  # Prédiction sur les données de test
            # Affichage des métriques de performance
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)  # Appel à la fonction pour afficher les graphiques sélectionnés

    # Configuration pour le modèle de régression logistique
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        # Paramètres ajustables pour la régression logistique dans la barre latérale
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step = 0.01, key = 'C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        # Bouton pour lancer la classification
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(x_train, y_train)  # Entraînement du modèle sur les données
            accuracy = model.score(x_test, y_test)  # Calcul de la précision du modèle
            y_pred = model.predict(x_test)  # Prédiction sur les données de test
            # Affichage des métriques de performance
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)  # Appel à la fonction pour afficher les graphiques sélectionnés

    # Configuration pour le modèle de forêt aléatoire
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        # Paramètres ajustables pour la forêt aléatoire dans la barre latérale
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        # Bouton pour lancer la classification
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(x_train, y_train)  # Entraînement du modèle sur les données
            accuracy = model.score(x_test, y_test)  # Calcul de la précision du modèle
            y_pred = model.predict(x_test)  # Prédiction sur les données de test
            # Affichage des métriques de performance
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)  # Appel à la fonction pour afficher les graphiques sélectionnés

    # Option pour afficher les données brutes
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)  # Affiche le DataFrame sur l'application web

# Vérifie si le script est exécuté directement et, dans ce cas, appelle la fonction main
if __name__ == '__main__':
    main()
