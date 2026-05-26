import streamlit as st
import pandas as pd
import os
import joblib
import gdown

MODEL_ID = st.secrets["GOOGLE_DRIVE_MODEL_ID"]
NOME_MODELO = "ModeloFinal_SMOTE_LightGBM.pkl"
NOME_PREPROCESSOR = "preprocessor.pkl" # Ele já estará na pasta raiz pelo GitHub

@st.cache_resource
def carregar_artefatos():
    # Baixa APENAS o Modelo do Drive se ele não existir
    if not os.path.exists(NOME_MODELO):
        with st.spinner("Baixando o modelo preditivo do Google Drive..."):
            gdown.download(id=MODEL_ID, output=NOME_MODELO, quiet=True)
            
    # O preprocessor não precisa de download, o Python lê direto da raiz do GitHub
    modelo = joblib.load(NOME_MODELO)
    preprocessor = joblib.load(NOME_PREPROCESSOR)
    return modelo, preprocessor

modelo, preprocessor = carregar_artefatos()
# Carrega os cérebros do projeto
modelo, preprocessor = carregar_artefatos()

# 2. Configuração Visual da Interface
st.set_page_config(page_title="Detector de Fraude IA 🛡️", page_icon="💳", layout="centered")

st.title("🛡️ Sistema Inteligente Antifraude")
st.markdown("""
    Esta inteligência artificial analisa o comportamento e o risco de transações financeiras 
    em plataformas de e-commerce utilizando um modelo otimizado de **LightGBM** com balanceamento **SMOTE**.
""")
st.write("---")

# 3. Formulário de Entrada de Dados para o Usuário
st.subheader("📋 Inserir Dados da Nova Transação")

with st.form("formulario_fraude"):
    col1, col2 = st.columns(2)
    
    with col1:
        # A variável se chama 'amt' (letras minúsculas)
        amt = st.number_input("Valor da Transação ($)", min_value=0.01, value=50.0)
        # A variável se chama 'gender'
        gender = st.selectbox("Gênero do Cliente", ["M", "F"])
        
    with col2:
        # A variável se chama 'category'
        category = st.selectbox("Categoria do Produto", ["shopping_net", "entertainment", "food_dining"])
        
    submetido = st.form_submit_button("⚡ Avaliar Risco de Fraude")

if submetido:
    # 1. Criamos o dicionário com TODAS as 13 colunas exatas que o modelo exige.
    # Mapeamos o que veio do formulário e o resto preenchemos com valores padrão (médias/frequentes).
    dados_usuario = pd.DataFrame([{
        # Colunas numéricas (usamos o valor do formulário ou a média estimada do dataset)
        "amount": amt,                               # O valor que o usuário digitou
        "account_age_days": 365,                     # Valor padrão neutro
        "shipping_distance_km": 15.0,                # Valor padrão neutro
        "avg_amount_user": amt,                      # Assumimos que é igual ao valor atual
        "total_transactions_user": 5,                # Valor padrão neutro
        "promo_used": 0,                             # 0 = Não usou promoção (padrão)
        
        # Colunas categóricas (mapeamos as do formulário e inventamos padrões para as outras)
        "merchant_category": category,               # A categoria que o usuário escolheu
        "gender": gender,                            # O gênero que o usuário escolheu
        "channel": "web",                            # Valor padrão estável
        "country": "US",                             # Valor padrão estável
        "bin_country": "US",                         # Valor padrão estável
        "three_ds_flag": "N",                        # N = Não tem 3D Secure (padrão)
        "avs_match": "Y",                            # Y = Bateu o endereço (padrão)
        "cvv_result": "M"                            # M = CVV Match/Correto (padrão)
    }])
    
    try:
        # 2. Agora o preprocessor receberá exatamente as colunas que ele espera e não vai quebrar!
        dados_tratados = preprocessor.transform(dados_usuario)
        
        # 3. Fazer a predição
        predicao = modelo.predict(dados_tratados)
        probabilidade = modelo.predict_proba(dados_tratados)[0][1]
        
        # 4. Apresentar o resultado
        st.write("### 📊 Resultado da Análise:")
        if predicao[0] == 1:
            st.error(f"🚨 **TRANSAÇÃO BLOQUEADA:** Alta suspeita de fraude!")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Crítico", delta_color="inverse")
        else:
            st.success(f"✅ **TRANSAÇÃO APROVADA:** Comportamento seguro.")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Seguro")
            
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {e}")
