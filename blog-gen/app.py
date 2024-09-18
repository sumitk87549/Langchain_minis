import streamlit as st
from decorator import decorate
from langchain.prompts import PromptTemplate
from langchain.llms import  ctransformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

model_path = "C:\\Users\\Sumit\\Desktop\\Langchain_Mini_projects\\Langchain_minis\\models\\llama3.1\\models--meta-llama--Meta-Llama-3.1-8B-Instruct\\snapshots\\5206a32e0bd3067aef1ce90f5528ade7d866253f"

# token = "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"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    # use_auth_token=token
)
# Load model
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.float16,  # Use float16 if your GPU supports it
#         device_map="auto",          # Adjust based on your hardware setup
#         # use_auth_token=token
#     )
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use float16 if your GPU supports it
    device_map="auto",
    low_cpu_mem_usage=True,
    # use_auth_token=token
)

model = load_checkpoint_and_dispatch(
    model, model_path, offload_state_dict=True, device_map="auto", offload_folder="E:\\tmpCheckpoints"
)

def generateBlogUsingLlama(input_prompt, word_limit, blog_type):
    # # Llama 3.1 model
    llm=ctransformers(model = model,
                      # model_type="llama",
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    # Prompt Template
    template=""" 
        Write a blog with following params :-
        Type = `{blog_type}`
        Input = `{input_prompt}`
        Word Limit = `{word_limit}`
        
    """
    prompt=PromptTemplate(input_variables=['blog_type', 'input_prompt', 'word_limit'],
                          template=template)
    # Generate response from llama 3 model
    response = llm(prompt.format(blog_style=blog_type, word_limit=word_limit, input_prompt=input_prompt))
    print(response)
    return response
    pass

    

# Functions to get response from LLama 3 model
st.set_page_config(page_title="Blog Gen", 
                   page_icon="🪄",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.header("Effortless Blogging 🪄")
st.caption("Turn Ideas into Articles")

input_text = st.text_input("Type it in, and let the words flow!")

# Creating 2 columns for 2 additional fields

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input('No. of Words')
with col2:
    blog_style = st.selectbox("Writing blog for: ", ('General','Research', 'AI', 'Fashion', 'Sports', 'Academics', 'Politics','Social', 'Travel','Culture', 'Outer Space', 'other'), index=0)
    
submit = st.button("Go!")

if submit:
    st.write(generateBlogUsingLlama(input_prompt=input_text, word_limit=no_words, blog_type=blog_style))
