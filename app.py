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
st.markdown(
    """
    Preencha os dados abaixo com as informações da transação que deseja avaliar.
    - **Valor da Transação:** valor total em dólares US$.
    - **Categoria do Produto:** escolha a categoria que melhor representa o item comprado.
    - **Dias desde o cadastro do cliente:** idade da conta do usuário em dias.
    - **Total de transações do usuário:** quantas compras o usuário já realizou.
    - **Distância do envio (km):** distância entre o endereço do cliente e o local de envio.
    - **Promoção / cupom usado:** indica se algum desconto ou cupom foi aplicado.
    - **3DS ativado:** indica se a autenticação 3D Secure foi usada na transação.
    - **AVS corresponde:** indica se o endereço informado bateu com o cadastro do cartão.
    - **Resultado do CVV:** resultado da verificação do código de segurança.
    """
)

with st.form("formulario_fraude"):
    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Valor da Transação ($)", min_value=0.01, value=50.0)
        category = st.selectbox("Categoria do Produto", ["shopping_net", "entertainment", "food_dining"])
        account_age_days = st.number_input("Dias desde o cadastro do cliente", min_value=0, value=365)
        total_transactions_user = st.number_input("Total de transações do usuário", min_value=0, value=5)

    with col2:
        shipping_distance_km = st.number_input("Distância do envio (km)", min_value=0.0, value=15.0)
        st.caption("Distância estimada entre a origem do envio e o destino final.")
        promo_used = st.selectbox("Promoção / cupom usado", ["Não", "Sim"])
        st.caption("Sim = desconto aplicado na compra; Não = preço normal.")
        three_ds_flag = st.selectbox("3DS ativado", ["N", "Y"])
        st.caption("Y = autenticação 3D Secure usada; N = não usada.")
        avs_match = st.selectbox("AVS corresponde", ["N", "Y"])
        st.caption("Y = endereço conferido pelo emissor do cartão; N = divergente.")
        cvv_result = st.selectbox("Resultado do CVV", ["M", "N", "U"])
        st.caption("M = código de segurança válido; N = inválido; U = não disponível.")

    submetido = st.form_submit_button("⚡ Avaliar Risco de Fraude")

if submetido:
    # 1. Criamos um dicionário com as colunas que o modelo realmente usa
    # O preprocessor foi treinado com estes campos já numéricos, então
    # precisamos converter os flags/códigos para números antes da transformação.
    dados_valores = {
        "amount": float(amt),
        "merchant_category": str(category),
        "bin_country": "US",
        "channel": "web",
        "country": "US",
        "three_ds_flag": 1.0 if three_ds_flag == "Y" else 0.0,
        "avs_match": 1.0 if avs_match == "Y" else 0.0,
        "cvv_result": 1.0 if cvv_result == "M" else (0.0 if cvv_result == "N" else 2.0),
        "account_age_days": float(account_age_days),
        "promo_used": 1.0 if promo_used == "Sim" else 0.0,
        "avg_amount_user": float(amt),
        "total_transactions_user": float(total_transactions_user),
        "shipping_distance_km": float(shipping_distance_km)
    }
    
    # 2. Descobrir a ordem exata que o preprocessor espera usando os metadados dele
    try:
        if hasattr(preprocessor, "feature_names_in_"):
            ordem_exata = list(preprocessor.feature_names_in_)
        else:
            # Caso o scikit-learn não tenha mapeado os nomes, usamos a lista de colunas esperada pelo pipeline
            ordem_exata = [
                "account_age_days", "total_transactions_user", "avg_amount_user", "amount",
                "country", "bin_country", "channel", "merchant_category", "promo_used",
                "avs_match", "cvv_result", "three_ds_flag", "shipping_distance_km"
            ]
        
        # Criamos o DataFrame garantindo apenas as colunas necessárias na ordem milimétrica correta
        dados_usuario = pd.DataFrame([{col: dados_valores[col] for col in ordem_exata}])
        
        # 3. Forçar a conversão de tipo coluna por coluna para alinhar com o pipeline
        colunas_numericas = [
            "amount", "account_age_days", "promo_used", "avg_amount_user", "total_transactions_user", 
            "shipping_distance_km", "three_ds_flag", "avs_match", "cvv_result"
        ]
        
        for col in dados_usuario.columns:
            if col in colunas_numericas:
                dados_usuario[col] = dados_usuario[col].astype(float)
            else:
                dados_usuario[col] = dados_usuario[col].astype(str)
                
    except KeyError as ke:
        st.error(f"Erro de mapeamento: O seu preprocessor espera uma coluna chamada {ke}, mas ela não foi incluída no app.py.")
        st.stop()

    try:
        # 4. Aplicar o pipeline de transformação
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
