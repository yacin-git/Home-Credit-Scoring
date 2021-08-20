import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
# import plotly_express as px

#Affichage des titres du Dashboard
st.title('Dashboard Prêt à dépenser - Scoring')
st.write("Visualisez et comparer les scores de vos clients !")

#Chargement des données
@st.cache
def load_data(file_name):
    data = joblib.load(file_name)
    return data
df = load_data("data_customers.pkl")


# Chargement du modèle
@st.cache(hash_funcs={'xgboost.sklearn.XGBClassifier': id})
def load_model(file_name):
    model = joblib.load(file_name)
    return model

XGBoost_model = load_model("best_model_XGBoost_pickle.pkl")

# Chargement des données SHAP
@st.cache(hash_funcs={'xgboost.sklearn.XGBClassifier': id})
def load_shap(df, model):
    df_shap = df.iloc[:,1:-2]
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    explainer_base_value = round(explainer.expected_value[0],3)
    shap_values = explainer.shap_values(df_shap)
    return explainer_base_value, shap_values

@st.cache
def parameters_waterfall(id_customer):
    df_customer_shap = df[df['SK_ID_CURR'] == id_customer].iloc[0,1:-2]
    index_customer = df[df['SK_ID_CURR'] == id_customer].iloc[:,1:-2].index
    shap_values_customer = shap_values[index_customer][0]
    return df_customer_shap, index_customer, shap_values_customer


#Mise en place des filtres
st.sidebar.title('Filtres')
# Filtre 1 GENDER
option1 = st.sidebar.selectbox('Sexe :',("Tous", "Homme", "Femme"))
sexe_filter = 1 if option1 == "Homme" else 0 if option1 == "Femme" else 2

# Filtre 2 NAME_CONTRACT_TYPE
option2 = st.sidebar.selectbox('Type de contrat :',("Tous", "Revolving loans", "Cash loans"))
contract_filter = 0 if option2 == "Revolving loans" else 1 if option2 == "Cash loans" else 2

# Filtre 3 AGE
max_age = round(max(-df['DAYS_BIRTH'])/365)
min_age = round(min(-df['DAYS_BIRTH'])/365)
min_slider, max_slider = st.sidebar.slider('Tranche d\'age', min_age, max_age, (min_age,max_age))

#Création du sous groupe de clients
df_group = df[["SK_ID_CURR", "NAME_CONTRACT_TYPE"
                  , "CODE_GENDER", "AMT_INCOME_TOTAL"
                  ,"CNT_CHILDREN","DAYS_BIRTH"
                  ,"SCORE","TARGET"]]
df_group = df_group[(df_group["DAYS_BIRTH"] < -min_slider*365) & (df_group["DAYS_BIRTH"] > -max_slider*365)]
df_group = df_group[df_group["CODE_GENDER"] != sexe_filter]
df_group = df_group[df_group["NAME_CONTRACT_TYPE"] != contract_filter]


#Sélection du client à étudier
st.sidebar.title('Sélectionnez un client')

# Filtre FINAL SK_ID_CURR
list_id = df_group['SK_ID_CURR'].unique().tolist()
id_customer = st.sidebar.selectbox('ID du client  :', list_id)
count_customers = df_group.shape[0]
# st.sidebar.write('Nombre de clients correspondant à vos filtres :', count_customers)
st.sidebar.write(count_customers, 'clients correspondant à vos filtres')

#Affichage des informations du client unique
df_customer = df[["SK_ID_CURR", "NAME_CONTRACT_TYPE"
                  , "CODE_GENDER", "AMT_INCOME_TOTAL"
                  ,"CNT_CHILDREN","DAYS_BIRTH"
                  ,"SCORE","TARGET"]]
df_customer = df_customer[df_customer['SK_ID_CURR'] == id_customer]


date = "Non renseigné" if len(df_customer['DAYS_BIRTH']) == 0 else round(-df_customer['DAYS_BIRTH'].item()/365)
name_type_contract = "Revolving loans" if df_customer["NAME_CONTRACT_TYPE"].item() == 1 else "Cash loans"
code_gender = "Femme" if df_customer["CODE_GENDER"].item() == 1 else "Homme"
cnt_children = df_customer["CNT_CHILDREN"].item()
amt_income_total = str(int(df_customer["AMT_INCOME_TOTAL"].item())) + " $"
score = str(round(df_customer["SCORE"].item()*100)) + "%"
target = "Non Eligible" if df_customer["TARGET"].item() == 1 else "Eligible"

st.write("ID client :", id_customer)
st.write("Sexe :", code_gender)
st.write("Age : " + str(date) + " ans")
st.write("Type de contrat :", name_type_contract)
st.write("Nombre d'enfants :", cnt_children)
st.write("Revenu total :", amt_income_total)
st.write("Probabilité de défaut :", score)
st.write("Statut du client :", target)

if target == "Eligible":
    st.write("[Voir les offres de crédits adaptées à ce client](https://homecredit.ph/all-about-loans/terms-and-conditions/)")
else:
    st.write("[Proposer des alternatives à ce client ?](https://homecredit.ph/tips-stories/sali-na-sa-loan-in-a-million-raffle-promo/)")
    


#Préparation de la visualitation SHAP
explainer_base_value, shap_values = load_shap(df, XGBoost_model)

#On trace le premier graph décrivant le client unique
df_customer_shap, index_customer, shap_values_customer = parameters_waterfall(id_customer)
fig1 = shap.waterfall_plot(shap.Explanation(values=shap_values_customer,
                                     base_values=explainer_base_value,
                                     data=df_customer_shap,
                                     feature_names=df.columns.tolist()),
                                     max_display=10)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig1)

#On trace le second graph décrivant le sous groupe similaire au client
checkbox_val = st.checkbox("Afficher la comparaison des " + str(count_customers) + " clients")
if checkbox_val:
    index_group = df_group.index
    shap_values_group = shap_values[index_group]
    fig2 = shap.summary_plot(shap_values_group, df.iloc[index_group,1:-2])
    st.pyplot(fig2)

