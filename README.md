# Detecção de Fraude em E-commerce 🛡️💳

Este repositório contém o desenvolvimento de uma solução de ponta a ponta em Ciência de Dados para a **identificação e mitigação de transações fraudulentas** em plataformas de e-commerce. O objetivo principal do projeto é analisar padrões de comportamento de compra, tratar o desbalanceamento severo dos dados reais de fraude e avaliar múltiplos algoritmos de Machine Learning para escolher o modelo preditivo ideal.

---

## 📌 Visão Geral do Problema

A fraude em e-commerce gera prejuízos bilionários anualmente, afetando a rentabilidade das empresas e a confiança dos consumidores. O principal desafio técnico na modelagem deste cenário é o **desbalanceamento extremo de classes**: a esmagadora maioria das transações é legítima (classe `0`), enquanto apenas uma fração mínima representa fraudes (classe `1`). 

Se um modelo for treinado sem o devido tratamento desse desbalanceamento, ele apresentará uma alta acurácia enganosa apenas por classificar tudo como legítimo, falhando criticamente na detecção do crime.

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

O projeto foi desenvolvido em **Python** dentro do ambiente Google Colab, utilizando as seguintes ferramentas:

* **Manipulação e Análise de Dados:** `pandas`, `numpy`
* **Visualização de Dados:** `matplotlib`, `seaborn`
* **Pré-processamento e Pipelines:** `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`)
* **Engenharia de Balanceamento:** `imbalanced-learn` (`SMOTE`, `ADASYN`, `RandomUnderSampler`)
* **Modelagem de Machine Learning:**
    * *Modelos Base:* Regressão Logística, Random Forest
    * *Modelos Avançados de Boosting:* XGBoost, LightGBM
* **Persistência do Modelo:** `joblib`

---

## 📐 Pipeline do Projeto

O fluxo de trabalho foi unificado e otimizado para evitar redundâncias de código e vazamento de dados (*data leakage*):

### 1. Preparação e Ingestão de Dados
* Conexão segura via API do Kaggle com criptografia de chaves através de variáveis de ambiente (`userdata`).
* Download automatizado do dataset `e-commerce-fraud-detection-dataset`.

### 2. Engenharia de Recursos & Pré-processamento
Para garantir que os dados entrassem nos modelos com a máxima qualidade, foi construído um pipeline automatizado usando `ColumnTransformer`:
* **Variáveis Numéricas:** Tratadas com `StandardScaler` para padronizar a escala (média 0 e desvio padrão 1), o que é crítico para o bom desempenho de algoritmos lineares como a Regressão Logística.
* **Variáveis Categóricas:** Transformadas via `OneHotEncoder(handle_unknown='ignore')` para converter categorias de texto em representações numéricas esparsas binárias, prevenindo erros ao encontrar novas categorias nos dados de teste.
* **Divisão de Dados:** Separação estrita em **80% para Treino** e **20% para Teste** antes de qualquer aplicação de balanceamento sintético.

### 3. Estratégias de Resampling (Tratamento de Desbalanceamento)
Foram testadas três abordagens competitivas de reamostragem no conjunto de treino:
1.  **SMOTE (Synthetic Minority Over-sampling Technique):** Cria amostras sintéticas da classe minoritária (fraude) ao longo da linha que une os vizinhos mais próximos.
2.  **ADASYN (Adaptive Synthetic Sampling):** Similar ao SMOTE, mas foca de forma adaptativa em gerar mais dados sintéticos nas regiões onde o modelo encontra maior dificuldade de separação (fronteiras de decisão complexas).
3.  **RandomUnderSampler:** Reduz aleatoriamente a quantidade de amostras da classe majoritária (legítima) até igualar à classe minoritária. Uma abordagem veloz, ideal para datasets massivos.

---

## 📊 Modelos Avaliados e Justificativas de Uso

Quatro algoritmos de diferentes complexidades foram colocados para competir sob as três técnicas de balanceamento:

| Modelo | Por que foi escolhido? |
| :--- | :--- |
| **Regressão Logística** | Serve como nosso *Baseline* estatístico. É linear, extremamente rápida de treinar e oferece excelente interpretabilidade de coeficientes. |
| **Random Forest** | Um modelo baseado em *Bagging* (árvores de decisão paralelas). Excelente para capturar interações não lineares entre variáveis sem sofrer com *overfitting* agressivo. |
| **XGBoost** | Algoritmo poderoso baseado em *Gradient Boosting*. Utiliza regularização L1/L2 interna para controlar a complexidade e costuma entregar métricas de ponta em dados tabulares esparsos. |
| **LightGBM** | Focado em alta performance e velocidade. Cresce de forma vertical (*leaf-wise*), o que o torna ideal para lidar com grandes volumes de transações e alta dimensionalidade de forma muito mais rápida que o XGBoost tradicional. |

---

## 📈 Critérios de Avaliação e Métricas

Na detecção de fraudes, a métrica de Acurácia comum não é confiável devido ao desbalanceamento. Por isso, a performance foi avaliada através do comportamento ponderado das seguintes métricas:

* **Precision (Precisão):** Das transações que o modelo classificou como fraude, quantas eram realmente fraude? Alta precisão reduz o número de *Falsos Positivos* (clientes legítimos bloqueados erroneamente).
* **Recall (Sensibilidade):** De todas as fraudes reais que aconteceram, quantas o modelo conseguiu capturar? Alto recall reduz os *Falsos Negativos* (fraudes que passaram despercebidas e geraram prejuízo financeiro).
* **F1-Score (Média Harmônica):** É o equilíbrio ideal entre Precision e Recall. Foi adotado como a **métrica de desempate (Métrica Vencedora)** para ordenar e escolher o modelo final do projeto.

---

## 🏆 Resultados e Automação de Seleção

O script executa um loop de matriz cruzada combinando cada **Técnica de Balanceamento** com cada **Modelo**, gerando um arquivo de relatório chamado `Relatorio_Balanceamento_Modelos.csv`.

O pipeline final extrai de forma 100% matemática a melhor combinação com base no maior `F1-score` ponderado. Após identificar os vencedores (ex: `LightGBM` combinado com `SMOTE`), o sistema:
1. Re-executa o balanceamento ideal apenas no set de treino.
2. Treina o classificador definitivo.
3. Exporta o artefato binário pronto para produção usando a biblioteca `joblib` com uma nomenclatura dinâmica (ex: `ModeloFinal_SMOTE_LightGBM.pkl`).

---

## 🚀 Como Executar o Projeto

1. Certifique-se de preencher as variáveis do Kaggle no ambiente seguro do seu notebook (`KAGGLE_USERNAME` e `KAGGLE_KEY`).
2. Execute as células em sequência.
3. O modelo final treinado será gerado no diretório raiz, acompanhado do relatório CSV com o desempenho detalhado de todos os cenários testados.

## Interfaz Grafica com Streamlit : [https://desafio-detecfraude-ecommerce.streamlit.app]
O usuario pode Avaliar riscos de fraude de transações financeiras em plataformas de e-commerce utilizando um modelo otimizado de LightGBM com balanceamento SMOTE.
Dados solicitados: 
Valor da Transação: valor total em dólares US$.
Categoria do Produto: escolha a categoria que melhor representa o item comprado.
Dias desde o cadastro do cliente: idade da conta do usuário em dias.
Total de transações do usuário: quantas compras o usuário já realizou.
Distância do envio (km): distância entre o endereço do cliente e o local de envio.
Promoção / cupom usado: indica se algum desconto ou cupom foi aplicado.
3DS ativado: indica se a autenticação 3D Secure foi usada na transação.
AVS corresponde: indica se o endereço informado bateu com o cadastro do cartão.
Resultado do CVV: resultado da verificação do código de segurança.
