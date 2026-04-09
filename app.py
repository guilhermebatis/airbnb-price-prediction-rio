# %%
import pandas as pd
import streamlit as st
import joblib
import os

@st.cache_resource
def carregar_modelo():
    if not os.path.exists('modelo/modelo_rf.joblib'):
        st.error("Modelo não encontrado. Certifique-se de que o arquivo 'modelo_rf.joblib' está na pasta 'modelo'.")
        return None
    return joblib.load('modelo/modelo_rf.joblib')


@st.cache_data
def carregar_colunas():
    dados = pd.read_csv('dados_tratado/dados.csv', sep=';')
    return list(dados.columns)[1:-1]

modelo = carregar_modelo()
colunas = carregar_colunas()

# ================= INPUTS =================

x_numericos = {
    'latitude': 0, 'longitude': 0, 'accommodates': 0,
    'bathrooms': 0, 'bedrooms': 0, 'beds': 0,
    'extra_people': 0, 'minimum_nights': 0,
    'ano': 0, 'mes': 0, 'n_amenities': 0,
    'host_listings_count': 0
}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_listas = {
    'property_type': ['Apartment','Bed and breakfast','Condominium','Guest suite','Guesthouse','Hostel','House','Loft','Outros','Serviced apartment'],
    'room_type': ['Entire home/apt','Hotel room','Private room','Shared room'],
    'cancellation_policy': ['flexible','moderate','strict','strict_14_with_grace_period']
}

features_categoricas = {}

# Numéricos
for item in x_numericos:
    if item in ['latitude', 'longitude']:
        valor = st.number_input(item, step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(item, step=0.01, value=0.0)
    else:
        valor = st.number_input(item, step=1, value=0)
    
    x_numericos[item] = valor

# Booleanos
for item in x_tf:
    valor = st.selectbox(item, ['Sim', 'Não'])
    x_tf[item] = 1 if valor == 'Sim' else 0

# One-hot
for item in x_listas:
    for valor in x_listas[item]:
        features_categoricas[f'{item}_{valor}'] = 0

for item in x_listas:
    valor = st.selectbox(item, x_listas[item])
    features_categoricas[f'{item}_{valor}'] = 1

# ================= PREDIÇÃO =================

if st.button('Prever preço'):
    features_categoricas.update(x_tf)
    features_categoricas.update(x_numericos)

    valores_x = pd.DataFrame(features_categoricas, index=[0])
    valores_x = valores_x.reindex(columns=colunas, fill_value=0)

    preco = modelo.predict(valores_x)




    st.success(f'Preço previsto: R$ {preco[0]:.2f}')