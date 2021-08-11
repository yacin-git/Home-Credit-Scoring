import streamlit as st
import pandas as pd
import joblib
import shap
# import plotly_express as px

#Affichage des titres du Dashboard
st.title('Dashboard Prêt à dépenser - Scoring')
st.write("Visualisez et comparer les scores de vos clients !")

#Chargement des données
@st.cache
def load_data(file_name):
    data = joblib.load(file_name)
    return data
df = load_data("data_customers_test.pkl")


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


# Filtre 1 SK_ID_CURR
list_id = df['SK_ID_CURR'].unique().tolist()
id_customer = st.sidebar.selectbox('ID du client  :', list_id)
st.sidebar.write('Nombre de clients disponibles :',df.shape[0])

#Affichage des informations du client
df_customer = df[["SK_ID_CURR", "NAME_CONTRACT_TYPE"
                  , "CODE_GENDER", "AMT_INCOME_TOTAL"
                  ,"CNT_CHILDREN"
                  ,"SCORE","TARGET"]]
df_customer = df_customer[df_customer['SK_ID_CURR'] == id_customer]

name_type_contract = "Revolving loans" if df_customer["NAME_CONTRACT_TYPE"].item() == 1 else "Cash loans"
code_gender = "F" if df_customer["CODE_GENDER"].item() == 1 else "M"
cnt_children = df_customer["CNT_CHILDREN"].item()
amt_income_total = str(df_customer["AMT_INCOME_TOTAL"].item()) + " $"
score = str(round(df_customer["SCORE"].item()*100)) + "%"
target = "Non Eligible" if df_customer["TARGET"].item() == 1 else "Eligible"

st.write("ID client :", id_customer)
st.write("Type de contrat :", name_type_contract)
st.write("Sexe :", code_gender)
st.write("Nombre d'enfants :", cnt_children)
st.write("Revenu total :", amt_income_total)
st.write("Probabilité de défaut :", score)
st.write("Statut du client :", target)


#Préparation de la visualitation SHAP
explainer_base_value, shap_values = load_shap(df, XGBoost_model)
# explainer = shap.TreeExplainer(XGBoost_model)

# graph = shap.summary_plot(shap_values, df_shap, plot_type="bar")
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.pyplot(graph)

#On recupère les valeurs du client en fonction de l'ID selectionné
df_customer_shap = df[df['SK_ID_CURR'] == id_customer].iloc[0,1:-2]
index_customer = df[df['SK_ID_CURR'] == id_customer].iloc[:,1:-2].index
shap_values_customer = shap_values[index_customer][0]

fig = shap.waterfall_plot(shap.Explanation(values=shap_values_customer,
                                     base_values=explainer_base_value,
                                     data=df_customer_shap,
                                     feature_names=df.columns.tolist()),
                                     max_display=10)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)



