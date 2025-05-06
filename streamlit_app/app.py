import streamlit as st
import requests

if "lyrics_input" not in st.session_state:
    st.session_state.lyrics_input = ""

def clear_input():
    st.session_state.lyrics_input = ""

st.title("Classificador de Gênero Musical 🎵")
st.write("Digite a letra da música abaixo:")

col1, col2 = st.columns(2)

with col2:
    if st.button("Limpar"):
        clear_input()

lyrics_input = st.text_area("Letra da música", key="lyrics_input")

with col1:
    if st.button("Classificar"):
        if not st.session_state.lyrics_input.strip():
            st.warning("Insira uma letra válida!")
        else:
            try:
                response = requests.post(
                    "https://lyrics-genre-classifier.onrender.com/predict",  # Se local: "http://localhost:8000/predict"
                    json={"lyrics": st.session_state.lyrics_input}
                )
                if response.status_code == 200:
                    genre = response.json()["genre"]
                    st.success(f"Gênero classificado: **{genre}**")
                else:
                    st.error("Erro na resposta da API")
            except Exception as e:
                st.error(f"Erro ao conectar com a API: {e}")



                 
