""" 
    Streamlit app for a chatbot using the TinyLlama model 
    Author: Wolf Paulus 
"""

from typing import List
import streamlit as st
import torch
from transformers import pipeline


@st.cache_resource
def load_model(model, dtype=torch.float16) -> pipeline:
    """ Loading the model """
    return pipeline("text-generation", model=model, torch_dtype=dtype, device_map="auto")


def init() -> None:
    """ Initialise session state variables """
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}]


def generate_response(user_input: str) -> None:
    """ Generate response, user_input: str - the user input """
    st.session_state['messages'].append(
        {"role": "user", "content": user_input})
    prompt = pipe.tokenizer.apply_chat_template(
        st.session_state['messages'], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True,
                   temperature=0.7, top_k=50, top_p=0.95)
    response = outputs[0]["generated_text"].split("<|assistant|>\n")[-1]
    st.session_state['messages'].append(
        {"role": "assistant", "content": response})


# Setting page title and header
st.set_page_config(page_title="Local Offline Bot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Local Offline Bot 🤖</h1>", unsafe_allow_html=True)

# Initialise session state variables
if not st.session_state.items():
    init()

# Load the model
model = "../TinyLlama-1.1B-Chat-v1.0"
dtype = torch.float16
pipe = load_model(model, dtype)
response_container = st.container()  # container for chat history
container = st.container()  # container for text box
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
    clear_button = st.button("Clear chat", on_click=init)
    if submit_button and user_input:
        generate_response(user_input)

with response_container:
    for i in range(1, len(st.session_state['messages'])):
        with st.chat_message(st.session_state['messages'][i]["role"]):
            st.write(st.session_state['messages'][i]["content"])
