# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# torch: the deep-learning framework the model runs on.
# transformers: Hugging Face library that gives us the model + tokenizer.
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

### Text formatting helpers
def format_property_value(property_name, value):
    """Format an RDKit descriptor for readable terminal output."""
    if property_name == "molecular_weight":
        return f"{value:.3f} g/mol"

    if property_name == "logP":
        return f"{value:.4f}"

    if property_name == "TPSA":
        return f"{value:.2f} Å²"

    # Count-based properties
    return str(int(value))


def print_properties_table(smiles, output_properties):
    """Print an aligned table using only built-in Python functionality."""
    headers = ["Property", "Value"]

    rows = [
        [
            property_data["property"],
            format_property_value(
                property_data["property"],
                property_data["value"],
            ),
        ]
        for property_data in output_properties
    ]

    if not rows:
        print(f"No properties could be calculated for {smiles}.")
        return

    column_widths = [
        max(
            len(headers[column_index]),
            *(len(row[column_index]) for row in rows),
        )
        for column_index in range(len(headers))
    ]

    header_line = " | ".join(
        headers[column_index].ljust(column_widths[column_index])
        for column_index in range(len(headers))
    )

    separator_line = "-+-".join(
        "-" * column_width
        for column_width in column_widths
    )

    print(f"Here are the calculated properties for {smiles}:")
    print(header_line)
    print(separator_line)

    for row in rows:
        print(
            " | ".join(
                row[column_index].ljust(column_widths[column_index])
                for column_index in range(len(row))
            )
        )

###

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
You are a chemical query parser.

Convert the user's request into valid JSON only. Do not answer the question,
calculate chemical properties, explain your reasoning, validate SMILES strings,
or use Markdown code fences.

Return exactly this JSON schema:

{
  "requests": [
    {
      "smiles": "exact SMILES string from the user",
      "properties": [
        "supported_property_name"
      ]
    }
  ],
  "missing_smiles": false,
  "unknown_request": false
}

The top-level JSON object must contain exactly these keys:

- requests
- missing_smiles
- unknown_request

missing_smiles and unknown_request must be JSON Boolean values: true or false.

Supported property names are exactly:

- molecular_weight
- logP
- TPSA
- hydrogen_bond_donors
- hydrogen_bond_acceptors
- rotatable_bonds
- rings
- aromatic_rings

Recognize equivalent wording and spelling variations:

- molecular weight, molecular wght, mol weight, mol. weight
  -> molecular_weight
- lipophilicity, hydrophobicity, logP, log P
  -> logP
- TPSA, topological polar surface area
  -> TPSA
- hydrogen bond donors, HBD
  -> hydrogen_bond_donors
- hydrogen bond acceptors, HBA
  -> hydrogen_bond_acceptors
- rotatable bonds
  -> rotatable_bonds
- rings, number of rings
  -> rings
- aromatic rings, number of aromatic rings
  -> aromatic_rings

Treat every chemical string as a SMILES string. Preserve it exactly as written,
including uppercase/lowercase letters, aromatic atoms, numbers, brackets, and
bond symbols. Do not convert it to a molecular formula, chemical name, or
different SMILES string.

Each SMILES string must appear only once in requests. Put all requested
properties for that SMILES string into its properties list.

For example:

User: Calculate the molecular weight and logP of CCO.

Return:

{
  "requests": [
    {
      "smiles": "CCO",
      "properties": [
        "molecular_weight",
        "logP"
      ]
    }
  ],
  "missing_smiles": false,
  "unknown_request": false
}

If the user requests a supported property but supplies no SMILES string, return:

{
  "requests": [],
  "missing_smiles": true,
  "unknown_request": false
}

If the user asks for an unsupported task, return:

{
  "requests": [],
  "missing_smiles": false,
  "unknown_request": true
}

If an unknown or invalid-looking chemical string is supplied with a supported
property, preserve it in requests exactly as written. Do not decide whether it
is a valid SMILES string; RDKit will validate it.
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


    # Parse the JSON response

    ## Is the response a valid JSON?

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        print("Error: The model's response is not valid JSON.")
        continue

    ## Does the JSON have the right structure?

    json_schema = {
        "requests": list,
        "missing_smiles": bool,
        "unknown_request": bool,
    }

    if not all(key in response_json for key in json_schema):
        print("Error: The model's response JSON does not have the expected keys.")
        continue

    ## Are request and smiles good?

    if response_json["unknown_request"]:
        print("Error: Invalid request detected")
        continue

    if response_json["missing_smiles"]:
        print("Error: SMILES string not detected")
        continue

    ## Get requests and make list for processing

    requests = response_json["requests"]

    if not isinstance(requests, list):
        print("Error: The 'requests' field is not a list.")
        continue

    ## Process each request

    PROPERTIES = ["molecular_weight", 
                  "logP", 
                  "TPSA", 
                  "hydrogen_bond_donors",
                  "hydrogen_bond_acceptors", 
                  "rotatable_bonds", 
                  "rings", 
                  "aromatic_rings"]


    # This loop goes through each request,
    # which has a single SMILES and a list
    # of properties
    for request in requests:
        properties = request.get("properties", [])
        output_properties = []

        smiles_string = request["smiles"]
        smiles_mol = Chem.MolFromSmiles(smiles_string)

        if smiles_mol is None:
            # This should be visible to the user
            print(f"Error: Invalid SMILES string '{smiles_string}'.")
            continue

        for property_name in properties:        
            if property_name not in PROPERTIES:
                print(f"Error: Unsupported property '{property_name}'.")
                continue

            property_value = None

            if property_name == "molecular_weight":
                property_value = Descriptors.MolWt(smiles_mol)
            elif property_name == "logP":
                property_value = Crippen.MolLogP(smiles_mol)
            elif property_name == "TPSA":
                property_value = rdMolDescriptors.CalcTPSA(smiles_mol)
            elif property_name == "hydrogen_bond_donors":
                property_value = Lipinski.NumHDonors(smiles_mol)
            elif property_name == "hydrogen_bond_acceptors":
                property_value = Lipinski.NumHAcceptors(smiles_mol)
            elif property_name == "rotatable_bonds":
                property_value = Lipinski.NumRotatableBonds(smiles_mol)
            elif property_name == "rings":
                property_value = Lipinski.RingCount(smiles_mol)
            elif property_name == "aromatic_rings":
                property_value = Lipinski.NumAromaticRings(smiles_mol)

            output_properties.append({
                "property": property_name,
                "value": property_value
            })

        print_properties_table(smiles_string, output_properties)

    # -----------------------------------------------------------------------
    # Do your work to do the tool calls based on LLM response
    # -----------------------------------------------------------------------
    
    # TODO 2: Based on the output of the LLM, make the appropriate tool
    # calls to RDKit and output the response in a user friendly manner.
    # Have fun in this part of the code!

