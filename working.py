import streamlit as st
import torch
from unsloth import FastLanguageModel

# Configuration parameters
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Define alpaca_prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Load FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="bsc-9/AI_Cooking_Kitchen",  # Replace with your trained model name
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Streamlit app interface
st.title("Chef Assistant Recipe Generator")

# User inputs
instruction = st.text_input("Instruction", "You are a professional chef assistant. Make a dish by firstly presenting it, then listing the ingredients and finally the recipe step by step in bullet points under the following input requirements.")
requirements = st.text_input("Requirements", "Requires 60 minutes preparation or less and it contains at least this ingredients chicken")

if st.button("Generate Recipe"):
    # Generate the recipe
    prompt = alpaca_prompt.format(instruction, requirements, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    recipe = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Display the recipe
    st.subheader("Generated Recipe")
    st.write(recipe)