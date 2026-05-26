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
    # 1. Criamos os dados com as 13 colunas originais do dataset
    dados_usuario = pd.DataFrame([{
        "amount": float(amt),
        "merchant_category": str(category),
        "gender": str(gender),
        "bin_country": "US",
        "channel": "web",
        "country": "US",
        "three_ds_flag": "N",
        "avs_match": "Y",
        "cvv_result": "M",
        "account_age_days": float(365),
        "promo_used": float(0),
        "avg_amount_user": float(amt),
        "total_transactions_user": float(5),
        "shipping_distance_km": float(15.0)
    }])
    
    # 2. ORDEM EXATA DAS COLUNAS (Alinhamento com o X_train do Colab)
    # O preprocessor falha se a ordem das colunas originais for alterada.
    # Esta lista abaixo reflete a ordem padrão esperada pelo seu pipeline:
    ordem_original_colunas = [
        "bin_country", "channel", "country", "three_ds_flag", "avs_match", 
        "cvv_result", "merchant_category", "amount", "account_age_days", 
        "promo_used", "avg_amount_user", "total_transactions_user", "shipping_distance_km"
    ]
    
    # Reordenamos o DataFrame de 1 linha para bater exatamente com o formato do treino
    dados_usuario = dados_usuario[ordem_original_colunas]
    
    # 3. Forçar explicitamente a conversão de tipos de cada coluna antes do transform
    colunas_numericas = ["amount", "account_age_days", "promo_used", "avg_amount_user", "total_transactions_user", "shipping_distance_km"]
    colunas_categoricas = ["bin_country", "channel", "country", "three_ds_flag", "avs_match", "cvv_result", "merchant_category", "gender"]
    
    # Garante que as numéricas existentes na lista sejam float e as de texto sejam string
    for col in dados_usuario.columns:
        if col in colunas_numericas:
            dados_usuario[col] = dados_usuario[col].astype(float)
        elif col in colunas_categoricas:
            dados_usuario[col] = dados_usuario[col].astype(str)
            
    try:
        # 4. Aplicar o pipeline de transformação sem erros de mapeamento posicional
        dados_tratados = preprocessor.transform(dados_usuario)
        
        # 5. Fazer a predição e calcular a probabilidade
        predicao = modelo.predict(dados_tratados)
        probabilidade = modelo.predict_proba(dados_tratados)[0][1]
        
        # 6. Apresentar o resultado visual na tela
        st.write("### 📊 Resultado da Análise:")
        if predicao[0] == 1:
            st.error(f"🚨 **TRANSAÇÃO BLOQUEADA:** Alta suspeita de atividade fraudulenta!")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Crítico", delta_color="inverse")
        else:
            st.success(f"✅ **TRANSAÇÃO APROVADA:** Comportamento seguro e legítimo.")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Seguro")
            
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {e}")
