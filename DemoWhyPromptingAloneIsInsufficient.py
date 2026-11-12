import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

# ---------------------------------------------------------------------------
# Model + device configuration
# ---------------------------------------------------------------------------
# The Hugging Face identifier for the model we're going to download and run.
model_id = "Qwen/Qwen3-4B"

# Pick the best available hardware to run the model on:
#   - "mps"  = Apple Silicon GPU (M1/M2/M3 Macs)
#   - "cuda" = NVIDIA GPU
#   - "cpu"  = fallback if neither GPU type is available
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print('Device: ', device)


tokenizer = AutoTokenizer.from_pretrained(model_id)

# Model: downloads the weights (first run only) and moves them onto `device`.
# Pick a precision that works on the current hardware:
#   - MPS (Apple Silicon): float16 (bfloat16 is not supported on MPS)
#   - CUDA (NVIDIA GPU):   float16 (works on all CUDA GPUs; saves memory)
#   - CPU (Windows/Linux/Mac without a GPU): float32 (half precision is
#     often slow or unsupported for many ops on CPU)
if device == "cpu":
    dtype = torch.float32
else:
    dtype = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)


prompt = input("\nPrompt: ")

# -----------------------------------------------------------------------
# Prepare the prompt for the model
# -----------------------------------------------------------------------
# 1) Build the chat as a list of role/content dicts. The system message
#    comes first so the model reads it before the user's question.
messages = [
    {"role": "system", "content": "Answer the question"},
    {"role": "user", "content": prompt},
]

# 2) Wrap the messages in the model's chat template (adds the special
#    role/turn tokens Qwen expects). `add_generation_prompt=True` appends
#    the marker that tells the model "your turn to reply."
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

# 3) Turn the formatted text into tensors of token IDs, and move them
#    onto the same device as the model.
inputs = tokenizer(text, return_tensors="pt").to(device)

# -----------------------------------------------------------------------
# Generate a response
# -----------------------------------------------------------------------
# `inference_mode` disables gradient tracking -> faster and uses less
# memory since we're only doing a forward pass, not training.
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256)

# -----------------------------------------------------------------------
# Decode and print just the newly generated text
# -----------------------------------------------------------------------
# `output` contains the prompt tokens followed by the generated tokens.
# Slicing from `inputs.input_ids.shape[1]:` skips the prompt so we only
# print what the model actually wrote. `skip_special_tokens=True` strips
# chat-template markers from the printed string.
response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print('LLM response: ', response)