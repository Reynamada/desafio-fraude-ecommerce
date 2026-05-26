import streamlit as st
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
    # Criando colunas para organizar o layout visual
    col1, col2 = st.columns(2)
    
    with col1:
        amt = st.number_input("Valor da Transação ($)", min_value=0.01, max_value=100000.0, value=50.0, step=0.5)
        gender = st.selectbox("Gênero do Cliente", ["M", "F"])
        
    with col2:
        category = st.selectbox("Categoria do Produto", [
            "shopping_net", "entertainment", "food_dining", 
            "gas_transport", "grocery_pos", "home", "kids_pets"
        ]) # Insira aqui as categorias exatas que existiam no seu dataset original
        
    # Se o seu modelo tiver mais variáveis numéricas (como idade, distância, etc.), adicione aqui:
    # idade = st.slider("Idade do Cliente", 18, 100, 35)

    # Botão de envio do formulário
    submetido = st.form_submit_button("⚡ Avaliar Risco de Fraude")

# 4. Lógica de Predição após o clique
if submetido:
    # 1. Construir o DataFrame com o formato EXATO esperado pelo ColumnTransformer
    dados_usuario = pd.DataFrame([{
        "amt": amt,
        "category": category,
        "gender": gender
        # Se houver mais colunas no X_train original, você DEVE incluí-las aqui na mesma ordem
    }])
    
    try:
        # 2. Aplicar o pipeline de escala (StandardScaler) e codificação (OneHotEncoder)
        dados_tratados = preprocessor.transform(dados_usuario)
        
        # 3. Fazer a predição e calcular a probabilidade
        predicao = modelo.predict(dados_tratados)
        probabilidade = modelo.predict_proba(dados_tratados)[0][1]
        
        # 4. Apresentar o resultado de forma visual impactante
        st.write("### 📊 Resultado da Análise:")
        
        if predicao[0] == 1:
            st.error(f"🚨 **TRANSAÇÃO BLOQUEADA:** Alta suspeita de atividade fraudulenta!")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Crítico", delta_color="inverse")
        else:
            st.success(f"✅ **TRANSAÇÃO APROVADA:** Comportamento seguro e legítimo.")
            st.metric(label="Risco Estimado", value=f"{probabilidade:.2%}", delta="Seguro")
            
    except Exception as e:
        st.error(f"Erro no processamento dos dados: {e}")
        st.info("Verifique se o formulário web contém todas as colunas estruturais que o modelo espera.")
