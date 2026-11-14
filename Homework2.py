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
You are a chemical-query parser for a Python/RDKit application.

Convert the user's request into valid JSON only. Do not answer the user,
calculate values, explain reasoning, validate SMILES, use Markdown, or include
text outside the JSON object.

Return exactly this schema:

{
  "requests": [
    {
      "smiles": "exact SMILES text from the user",
      "properties": [
        "supported_property_name"
      ]
    }
  ],
  "missing_smiles": false,
  "unknown_request": false
}

The top-level object must contain exactly these keys:

- requests
- missing_smiles
- unknown_request

missing_smiles and unknown_request must be JSON Booleans: true or false.

Supported property names are exactly:

- molecular_weight
- logP
- TPSA
- hydrogen_bond_donors
- hydrogen_bond_acceptors
- rotatable_bonds
- rings
- aromatic_rings

Property-language mapping:

- molecular weight, mol weight, mol. weight, molecular wght
  -> molecular_weight
- logP, log P, lipophilicity, hydrophobicity
  -> logP
- TPSA, topological polar surface area
  -> TPSA
- hydrogen bond donors, hydrogen-bond donors, HBD
  -> hydrogen_bond_donors
- hydrogen bond acceptors, hydrogen-bond acceptors, HBA
  -> hydrogen_bond_acceptors
- rotatable bonds
  -> rotatable_bonds
- rings, ring count, number of rings
  -> rings
- aromatic rings, aromatic ring count
  -> aromatic_rings

Treat chemical-looking strings as SMILES strings. Preserve valid SMILES characters 
exactly. Ignore surrounding conversational punctuation, such as trailing ?, !, 
commas, and sentence-ending periods.

Do not:
- Interpret SMILES as molecular formulas.
- Rename, repair, or validate SMILES.
- Convert lowercase aromatic SMILES into uppercase SMILES.
- Infer additional properties.
- Calculate property values.

Association rules:

- Each SMILES appears only once in requests.
- Each request contains one SMILES and every property explicitly requested for
  that specific SMILES.
- Do not assign all properties to every SMILES.
- Associate a property only with the SMILES named alongside it in the user’s
  wording.
- If one property applies to molecule A and another property applies to
  molecule B, create separate request objects.
- If several properties apply to one SMILES, place all of them in that
  SMILES's properties list.
- A request for "all eight properties" means all eight supported properties.

Pattern example: different properties for different molecules

User:
"Find the mass of CCN, then the polar surface area and ring count of O=C1NCC1."

Return:

{
  "requests": [
    {
      "smiles": "CCN",
      "properties": [
        "molecular_weight"
      ]
    },
    {
      "smiles": "O=C1NCC1",
      "properties": [
        "TPSA",
        "rings"
      ]
    }
  ],
  "missing_smiles": false,
  "unknown_request": false
}

Pattern example: spelling errors and noisy punctuation

User:
"mol wght::: CCN!!! and hydrophobcity of O=C=O???"

Return:

{
  "requests": [
    {
      "smiles": "CCN",
      "properties": [
        "molecular_weight"
      ]
    },
    {
      "smiles": "O=C=O",
      "properties": [
        "logP"
      ]
    }
  ],
  "missing_smiles": false,
  "unknown_request": false
}

Missing-SMILES rules:

If the user requests at least one supported property but supplies no SMILES
string, return an empty requests list with missing_smiles true and
unknown_request false.

Never include a request object with an empty smiles string.

Pattern example:

User:
"Can you report HBA and aromatic ring count?"

Return:

{
  "requests": [],
  "missing_smiles": true,
  "unknown_request": false
}

Unknown-request rules:

Set unknown_request to true only if the user requests no supported property.
For an unknown request, return an empty requests list and set missing_smiles
to false.

Pattern example:

User:
"What is the melting temperature of CCN?"

Return:

{
  "requests": [],
  "missing_smiles": false,
  "unknown_request": true
}

If the user supplies an invalid-looking chemical string while requesting a
supported property, preserve it exactly in requests. RDKit will determine
whether it is valid.

Return valid JSON only.
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

