import streamlit as st

from utils import scrapper



import random
import time

#-------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
READER_MODEL_NAME  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


#-------------------------------------------------------------------
@st.cache_resource
def init():
    print('init')
    # scrap articles from the last 24h from Le Monde
    
    return 0

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

#-------------------------------------------------------------------
#--------------------------- APP -----------------------------------
#-------------------------------------------------------------------

st.set_page_config(
    page_title="RT News Summarizer",
    layout="centered",
    #layout="wide",
    initial_sidebar_state="expanded")

init()

st.title("Realtime News Summarizer")

st.write('Answers will be based on articles from [Le Monde](https://www.lemonde.fr) on the last 24h.')
st.write(f'Current LLM is : [{READER_MODEL_NAME}](https://huggingface.co/{READER_MODEL_NAME})')

st.header('News subjects :')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter a subject you would like news on"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- Do magic here ---
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})