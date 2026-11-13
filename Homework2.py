# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# torch: the deep-learning framework the model runs on.
# transformers: Hugging Face library that gives us the model + tokenizer.
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

# ---------------------------------------------------------------------------
# System prompt  <-- EDIT ME to change the model's persona / behavior
# ---------------------------------------------------------------------------
# The system prompt is sent to the model before every user question. Use it
# to set tone, role, constraints, output format, etc.

# TODO 1: Write a system prompt here giving instructions to the 
# AI model on how to handle the user's input.
SYSTEM_PROMPT = '''
You are a chemical query parser. You do not answer the user's question and you do not calculate chemical properties.

Convert the user's request into valid JSON only.

The supported properties are:
- molecular_weight
- logP
- TPSA
- hydrogen_bond_donors
- hydrogen_bond_acceptors
- rotatable_bonds
- rings
- aromatic_rings

Treat every chemical string as a SMILES string, not as a molecular formula.
Preserve SMILES exactly, including lowercase aromatic atoms such as c1ccccc1.
Do not reinterpret, rename, or correct a SMILES string.
Recognize spelling variations such as:
- molecular weight, mol weight, mol. weight
- logP, lipophilicity, hydrophobicity
- TPSA, topological polar surface area
- hydrogen bond donors, HBD
- hydrogen bond acceptors, HBA
- rotatable bonds
- rings
- aromatic rings

Return one request object for every property-and-SMILES pair.
If the user asks for multiple properties or molecules, return multiple request objects.
If a supported property is requested without a SMILES string, set missing_smiles to true.
If the request is unrelated to the supported properties, set unsupported_request to true.
Do not calculate values.
Do not provide explanations.
Return JSON only using this schema:

{
  "requests": [
    {
      "property": "one_supported_property",
      "smiles": "exact_smiles_from_user"
    }
  ],
  "unsupported_request": true_or_false,
  "missing_smiles": true_or_false
}
'''

# ---------------------------------------------------------------------------
# Load the tokenizer and model
# ---------------------------------------------------------------------------
# Tokenizer: converts text <-> token IDs the model understands.
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

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
# Keep asking the user for prompts until they type "quit" or "exit"
# (or press Ctrl+C). Each turn is independent — the model does NOT remember
# previous questions in this loop, since we rebuild `messages` from scratch
# every iteration.
while True:
    prompt = input("\nPrompt (type 'quit' to exit): ")
    if prompt.strip().lower() in {"quit", "exit", "q"}:
        break

    # -----------------------------------------------------------------------
    # Prepare the prompt for the model
    # -----------------------------------------------------------------------
    # 1) Build the chat as a list of role/content dicts. The system message
    #    comes first so the model reads it before the user's question.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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

    # TODO 1.5: Make sure you delete this line of code before submitting.
    # This line of code is simply here to help you to debug.
    print('LLM response so far: ', response) 

    # -----------------------------------------------------------------------
    # Do your work to do the tool calls based on LLM response
    # -----------------------------------------------------------------------
    
    # TODO 2: Based on the output of the LLM, make the appropriate tool
    # calls to RDKit and output the response in a user friendly manner.
    # Have fun in this part of the code!

