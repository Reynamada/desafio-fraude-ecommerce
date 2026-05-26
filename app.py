import streamlit as st

st.title("Teste de Inicialização do Servidor")
st.success("Se você está vendo esta mensagem, o Streamlit Cloud está funcionando perfeitamente!")

# Bloco para testar se os segredos estão mapeados
try:
    st.write("Verificando Secrets...")
    if "GOOGLE_DRIVE_MODEL_ID" in st.secrets:
        st.write("✅ ID do Modelo encontrado nos Secrets!")
    else:
        st.warning("⚠️ ID do Modelo NÃO encontrado nos Secrets.")
except Exception as e:
    st.error(f"Erro ao acessar os Secrets: {e}")
