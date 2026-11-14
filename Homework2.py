# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# torch: the deep-learning framework the model runs on.
# transformers: Hugging Face library that gives us the model + tokenizer.
import json
import agent_functions

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

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
SYSTEM_PROMPT = """
You are a parser for a chemical-property application. Return valid JSON only;
never include Markdown, explanations, calculations, or text outside the JSON.
Do not validate, rename, repair, or calculate properties for SMILES strings.

The response must contain exactly these keys:
{
  "requests": [{"smiles": "SMILES", "properties": ["property_name"]}],
  "missing_smiles": false,
  "unknown_request": false
}

Allowed property names:
molecular_weight, logP, TPSA, hydrogen_bond_donors,
hydrogen_bond_acceptors, rotatable_bonds, rings, aromatic_rings.

Map natural language to those names:
molecular weight, mol weight, mol. weight, and molecular wght mean
molecular_weight. logP, log P, lipophilicity, and hydrophobicity mean logP.
TPSA and topological polar surface area mean TPSA. HBD means
hydrogen_bond_donors; HBA means hydrogen_bond_acceptors. Recognize common
spelling errors. Preserve lowercase aromatic atoms in SMILES. Ignore surrounding
conversational punctuation such as trailing question marks and exclamation marks.

First classify the request, before extracting any SMILES:

- If no allowed property was requested, return exactly:
  {"requests": [], "missing_smiles": false, "unknown_request": true}

- If one or more allowed properties were requested but no non-empty SMILES was
  supplied, return exactly:
  {"requests": [], "missing_smiles": true, "unknown_request": false}

- Otherwise, return the requested SMILES/property pairs with both Boolean
  values false.

A request for an allowed property is not unknown merely because it has no
SMILES, molecule name, or formula. Conversely, a SMILES string by itself does
not make an unsupported request missing_smiles: if no allowed property was
requested, unknown_request must be true even when a SMILES is present. When
missing_smiles is true, the complete value of requests must be exactly []:
do not include an object, an empty smiles string, or a property name.

Example of a supported request with no SMILES:
Input: "Report the hydrogen-bond acceptor count."
Output: {"requests": [], "missing_smiles": true, "unknown_request": false}

Input: "Provide molecular weight information."
Output: {"requests": [], "missing_smiles": true, "unknown_request": false}

Input: "Give the ring count."
Output: {"requests": [], "missing_smiles": true, "unknown_request": false}

Example of an unsupported request with a SMILES:
Input: "Describe the odor of CO."
Output: {"requests": [], "missing_smiles": false, "unknown_request": true}

Input: "Is C#N soluble in water?"
Output: {"requests": [], "missing_smiles": false, "unknown_request": true}

Each SMILES must appear once. Put all properties requested for that exact
SMILES in its properties list. Do not assign every property to every SMILES;
associate each property only with the molecule specified for it. A request for
all eight properties includes every allowed property.

If a supported property has an invalid-looking SMILES, preserve that string
exactly. RDKit will validate it later.
"""

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
    
    model.eval()
    
    with torch.inference_mode():
        output = model.generate(
                    **inputs, 
                    max_new_tokens=160,
                    do_sample=False,
                    use_cache=True
                    )

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

    validation_error_message = agent_functions.json_validation(response)

    if validation_error_message:
        print(validation_error_message)
        continue

    requests = json.loads(response)["requests"]

    output_properties = agent_functions.process_requests(requests)

    agent_functions.print_properties_tables(output_properties)

    # -----------------------------------------------------------------------
    # Do your work to do the tool calls based on LLM response
    # -----------------------------------------------------------------------
    
    # TODO 2: Based on the output of the LLM, make the appropriate tool
    # calls to RDKit and output the response in a user friendly manner.
    # Have fun in this part of the code!
