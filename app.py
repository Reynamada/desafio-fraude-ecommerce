import streamlit as st
import pandas as pd
import joblib
import os
import requests

# 1. Configuração do ID do arquivo e nome local
ID_DO_ARQUIVO = "GOOGLE_DRIVE_MODEL_ID"  # Substitua pelo ID real do seu arquivo
NOME_ARQUIVO_LOCAL = "ModeloFinal_SMOTE_LightGBM.pkl"

@st.cache_resource
def baixar_e_carregar_modelo():
    # Se o modelo não estiver na máquina do servidor do Streamlit, faz o download
    if not os.path.exists(NOME_ARQUIVO_LOCAL):
        with st.spinner("Baixando o modelo preditivo do Google Drive... Por favor, aguarde."):
            # URL de download direto para arquivos públicos do Google Drive
            url = f"https://docs.google.com/uc?export=download&id={ID_DO_ARQUIVO}"
            
            # Realiza a requisição e salva o arquivo localmente
            resposta = requests.get(url, stream=True)
            if resposta.status_code == 200:
                with open(NOME_ARQUIVO_LOCAL, "wb") as f:
                    for chunk in resposta.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Download do modelo concluído com sucesso!")
            else:
                st.error("Erro ao baixar o modelo. Verifique as permissões do link do Google Drive.")
                return None

    # Carrega o modelo usando o joblib
    modelo = joblib.load(NOME_ARQUIVO_LOCAL)
    return modelo

# Chamar a função para carregar o modelo na aplicação
modelo = baixar_e_carregar_modelo()
