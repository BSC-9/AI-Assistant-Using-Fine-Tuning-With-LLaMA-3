!pip install pyarrow==14.0.1 requests==2.31.0
!pip install streamlit
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes

with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import torch
from unsloth import FastLanguageModel

max_seq_length = 2048
dtype = None
load_in_4bit = True

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

@st.cache_resource(show_spinner=False)
def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="bsc-9/AI_Cooking_Kitchen",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model) 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_model()

name = st.text_input("Enter your name", "Chef")
st.title(f"{name}'s Assistant Recipe Generator")
instruction = st.text_input("Instruction", "You are a professional chef assistant. Make a dish by firstly presenting it, then listing the ingredients and finally the recipe step by step in bullet points under the following input requirements.")
requirements = st.text_input("Requirements", "Requires 60 minutes preparation or less and it contains at least this ingredients chicken biryani")

if st.button("Generate Recipe"):
    # Generate the recipe
    prompt = alpaca_prompt.format(instruction, requirements, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    recipe = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    st.subheader("Generated Recipe")
    st.write(recipe + "\\n\\nThank you for making me your cooking assistant!")
''')


import subprocess
import time
!kill -9 $(lsof -t -i:8501)
time.sleep(5)
!streamlit run app.py & npx localtunnel --port 8501
