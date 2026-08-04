import os
import torch

# Optimize PyTorch CPU threading
if not torch.cuda.is_available():
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"PyTorch configured for CPU with {torch.get_num_threads()} threads")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Primary fine-tuned model configuration
MODEL_ID = "dharaamehta33/medical-chatbot-llama"

# Global variables for model and tokenizer
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the fine-tuned model ONCE at startup.
    No fallbacks or stubs: raises real exceptions if model loading fails.
    """
    global model, tokenizer
    print(f"Starting model load for HF repository: {MODEL_ID}")
    hf_token = os.getenv("HF_TOKEN", None)

    try:
        print(f"Attempting to read PEFT adapter config from {MODEL_ID}...")
        peft_config = PeftConfig.from_pretrained(MODEL_ID, token=hf_token)
        base_model_id = peft_config.base_model_name_or_path
        print(f"Detected PEFT adapter. Loading base model: {base_model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )
        print(f"Applying PEFT adapter from {MODEL_ID}...")
        model = PeftModel.from_pretrained(base_model, MODEL_ID, token=hf_token)
    except Exception as peft_err:
        print(f"PEFT load returned: {peft_err}. Trying direct AutoModel load...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            token=hf_token
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded successfully")
    yield
    print("Shutting down application...")

# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Medical Q&A Chatbot API",
    description="Backend API powered by fine-tuned Llama 3.2 1B model (dharaamehta33/medical-chatbot)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading or failed to initialize.")
    
    user_q = request.question.strip()

    # Fast response for simple greetings
    if user_q.lower() in ["hi", "hello", "hey", "hi!", "hello!", "hey!"]:
        return {"answer": "Hello! I am your AI Medical Assistant. How can I help you with your health questions today?"}

    try:
        system_instruction = (
            "You are an expert AI medical assistant. Provide clear, accurate, and helpful answers "
            "to medical questions while maintaining clinical safety standards."
        )
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_q}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Define EOS stop tokens to halt generation immediately when complete
        eos_ids = [tokenizer.eos_token_id]
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id is not None and isinstance(eot_id, int) and eot_id > 0:
                eos_ids.append(eot_id)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids
            )
            
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = output_ids[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if not answer:
            answer = "I apologize, but I could not generate a response. Please consult a qualified healthcare professional."
            
        return {"answer": answer}

    except Exception as e:
        print(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
