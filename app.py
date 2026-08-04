import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load model from HuggingFace Hub
BASE = "unsloth/Llama-3.2-1B-Instruct"
LORA = "dharaamehta33/medical-chatbot-llama"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(base_model, LORA)
model.eval()
print("Model ready!")

# Inference Prompt
PROMPT = """Below is an instruction paired with input. Write a response.

### Instruction:
You are a helpful medical assistant. Answer clearly and safely.

### Input:
{symptoms}

### Response:
"""

def get_advice(symptoms):
    if not symptoms or not symptoms.strip():
        return "Please describe your symptoms."

    q_clean = symptoms.strip().lower()
    if q_clean in ["hi", "hello", "hey", "hi!", "hello!"]:
        return "Hello! I am your AI Medical Assistant. How can I help you with your health questions today?"

    prompt = PROMPT.format(symptoms=symptoms.strip())
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        return response.split("### Response:")[-1].strip()
    return response.strip()

# Gradio UI
with gr.Blocks(title="AI Medical Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 AI Medical Symptom Checker")
    gr.Markdown("**Powered by Llama 3.2 + Unsloth Fine-tuning**")
    gr.Markdown("⚠️ *For educational demo only. Not real medical advice.*")
    
    with gr.Row():
        with gr.Column():
            symptoms_input = gr.Textbox(
                label="Describe Your Symptoms",
                placeholder="e.g. I have fever, cough, and body pain...",
                lines=3
            )
            submit_btn = gr.Button("Get Advice", variant="primary")
        with gr.Column():
            output = gr.Textbox(label="AI Medical Advice", lines=8)
            
    submit_btn.click(get_advice, inputs=symptoms_input, outputs=output)
    symptoms_input.submit(get_advice, inputs=symptoms_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
