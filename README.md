# Music Genre Classifier 🎵

Este projeto tem como objetivo classificar letras de músicas brasileiras em quatro gêneros distintos: Bossa Nova, Funk, Gospel e Sertanejo.

Teste agora mesmo: https://lyricsgenreclassifier.streamlit.app/

## 🔍 Etapas do Projeto

### 1. Coleta e Leitura dos Dados
Foram utilizados quatro arquivos CSV localizados na pasta `data/`, contendo letras de músicas por gênero.

### 2. Pré-processamento e limpeza
- Tokenização e limpeza (remoção de stopwords, pontuação, letras minúsculas): A motivação se deu, para melhorar a eficiência do modelo através da vetorização, que não atribui pouco ou nenhum valor em palavras do tipo.
- Foi criada uma nova variável, a partir da tokenização, para ser usada como preditora no modelo.
- Remoção de campos vazios.

### 2. Análise Exploratória (EDA)
- Distribuição de músicas por gênero
- Comprimento médio das letras
- Frequência de palavras por gênero
- Wordclouds
- Top 10 palavras por gênero
- Número de palavras únicas por gênero
- Histogramas das distribuições
- Na análise exploratório foi possível avaliar também o balanceamento entre as classes e notas comportamentos diferentes entre os gêneros.

### 4. Modelagem e Avaliação
- Separação treino/teste (80/20)
- Vetorização usando `TfidfVectorizer`: transforma os textos (strings) em vetores numéricos, usando o método TF-IDF (Term Frequency-Inverse Document Frequency), que pesa a importância das palavras em relação ao corpus.
- Modelos testados:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - LinearSVC
- Validação cruzada (StratifiedKFold) e Otimização de hiperparâmetros via GridSearchCV: Para ser possível treinar os modelos com vários parâmetros e fatias diferente do dataset, com o objetivo de não enviesar o modelo antes de aplica-lo ao conjunto de teste.
- Métrica principal: Acurácia - dado que as classes estão balanceadas.
- Outras métricas de suporte foram utilizadas, como Precision, Recall e F1 - além da matriz de confusão.
- Escolha do modelo final e salvamento do arquivo cerealizado.

### 5. Deploy da Solução

#### API REST (FastAPI)
O modelo final foi disponibilizado via API usando FastAPI. Essa API pode ser executada localmente:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

- Mas também foi implementada em produção através da plataforma https://render.com/

#### Interface Web (Streamlit)
Uma interface interativa foi criada com Streamlit, permitindo ao usuário digitar a letra de uma música e receber a classificação. Esta interface:
- Pode ser executada localmente:
```bash
streamlit run app.py
```
- E foi publicado em deploy no endereço: https://lyricsgenreclassifier.streamlit.app/
- É esperado uma lentidão (50s) na primeira execução, dado que o render fica em standby.

## ⚙️ Como Executar Localmente

1. Clone o repositório
```bash
git clone https://github.com/feliperastelli/lyrics_genre_classifier.git
cd lyrics_genre_classifier
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Treine o modelo (opcional, já fornecido em `models/model.pkl`)
```bash
python train_model.py
```

4. Inicie a API:
```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

5. Execute a interface Streamlit:
   - Antes de executar a interface, altere o host para `http://localhost:8000/predict` para que leia o modelo salvo localmente e não o que foi disponibilizado em Cloud
```bash
streamlit run app.py
```

## ✅ Resultados
A solução apresentou boa performance de classificação, com destaque para os modelos `Logistic Regression` e `LinearSVC`. O modelo final foi salvo em `models/model.pkl`.

## 📦 Estrutura do Projeto
```
.
├── api_model/
│   └── __init__.py         # Arquivo informativo
│   └── main.py             # API FastAPI
│   └── model.py            # Pipeline de treinamento
│   └── requirements.txt      
├── streamlit_app/
│   └── app.py              # Interface Streamlit
│   └── requirements.txt    
├── models/
│   └── model.pkl           # Modelo final treinado
├── data/
│   ├── bossa_nova.csv
│   ├── funk.csv
│   ├── gospel.csv
│   └── sertanejo.csv
├── utils.py                # Funções auxiliares de preprocessamento
├── render.yaml             # Configurações para o deploy da API
├── requirements.txt
└── README.md
```
