import streamlit as st
import pandas as pd
import joblib
import os
import requests
import gdown

# Busque o ID do seu segredo configurado no Streamlit Cloud
ID_DO_ARQUIVO = st.secrets["GOOGLE_DRIVE_MODEL_ID"]
NOME_ARQUIVO_LOCAL = "ModeloFinal_SMOTE_LightGBM.pkl"

@st.cache_resource
def baixar_e_carregar_modelo():
    # Verifica se o modelo já não foi baixado para não repetir o processo
    if not os.path.exists(NOME_ARQUIVO_LOCAL):
        with st.spinner("Baixando o modelo preditivo do Google Drive... Isso pode levar alguns segundos devido ao tamanho do arquivo."):
            
            # Monta a URL de download que o gdown precisa
            url = f"https://drive.google.com/uc?id={ID_DO_ARQUIVO}"
            
            try:
                # O gdown baixa o arquivo e lida com o aviso de vírus automaticamente
                gdown.download(url, NOME_ARQUIVO_LOCAL, quiet=False)
                st.success("Download do modelo concluído com sucesso!")
            except Exception as e:
                st.error(f"Erro crítico ao baixar o modelo: {e}")
                st.info("Verifique se o ID nos Secrets está correto e se o arquivo está público para 'Qualquer pessoa com o link'.")
                return None
                
    # Carrega o modelo na memória
    return joblib.load(NOME_ARQUIVO_LOCAL)

# Ativa a função
modelo = baixar_e_carregar_modelo()
