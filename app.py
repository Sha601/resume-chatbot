import streamlit as st
import json
import os
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.hf import hf_embed
from transformers import AutoModel, AutoTokenizer
import requests


# Import custom LLM function
import asyncio

async def mistral_llm(prompt, llm_call, system_prompt=None, history_messages=[], temperature=0, **kwargs):
    URL = 'http://crmdi-gpu4:5000/gpu/llm/text/api/llm_generation/generate'
    SIGNATURE = 'YOUR_SIGNATURE_HERE'
    content_type = 'application/json'

    message = '<s>'
    if system_prompt:
        message += f'[INST] {system_prompt} [/INST]'

    for h_message in history_messages:
        content = h_message['content']
        message += f'\n\n[INST] {content} [/INST]'

    message += f'\n\n[INST] {prompt} [/INST]'

    data = {
        'prompt': f'<s>[INST] {message} [/INST]',
        'max_tokens': 5000,
        'temperature': temperature
    }
    json_data = json.dumps(data)

    response = requests.post(url=URL, data=json_data, headers={'signature': SIGNATURE, 'Content-Type': content_type})
    result = json.loads(response.text)['text'][0]
    return result

# async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
#     return await mistral_llm(prompt=prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    llm_call = kwargs.get("llm_call", None)  # Ensure llm_call is passed
    return await mistral_llm(prompt=prompt, llm_call=llm_call, system_prompt=system_prompt, history_messages=history_messages, **kwargs)

# Define Custom Model Class
class CustomModel:
    def __init__(self, project_path):
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        self.project_path = project_path
        self.rag = LightRAG(
            working_dir=project_path,
            llm_model_name="mistral_llm",
            chunk_token_size=1500,
            chunk_overlap_token_size=100,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=5000,
                func=lambda texts: hf_embed(
                    texts,
                    tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
                    embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                )
            )
        )

    def train(self, data, custom_kg: bool = True):
        if custom_kg:
            self.rag.insert_custom_kg(data)
        else:
            if not isinstance(data, str):
                raise ValueError("Text data must be a string when custom_kg is False.")
            text_data = data
            self.rag.insert(text_data)
        st.success("Model trained successfully!")

    def query(self, query_text: str, mode="local"):
        response = self.rag.query(query_text, param=QueryParam(mode=mode))
        return response

# Streamlit UI
st.set_page_config(page_title="Resume Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Resume Chatbot")

# Initialize model
PROJECT_PATH = "./resume_chatbot_final"
model = CustomModel(project_path=PROJECT_PATH)

# File upload
st.sidebar.header("Upload Resume Data")
uploaded_file = st.sidebar.file_uploader("Upload a JSON file for training", type=["json"])

if uploaded_file is not None:
    data = json.load(uploaded_file)
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
             model.train(data, custom_kg=True)
        st.sidebar.success("Training completed!")

# Query Section
st.header("Candidate Search")

query_text = st.text_input("Enter your query:", "List the candidates who have current role as HR.")
query_mode = st.selectbox("Select Query Mode:", ["local", "global", "hybrid", "mix"])

if st.button("Search Candidates"):
    if query_text:
        with st.spinner("Fetching results..."):
            response = model.query(query_text, mode=query_mode)
        st.subheader("Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")
