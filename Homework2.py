# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# torch: the deep-learning framework the model runs on.
# transformers: Hugging Face library that gives us the model + tokenizer.
import argparse
import json
import logging
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import agent_functions

import torch
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging
# from rdkit import Chem
# from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

parser = argparse.ArgumentParser(description="Chemical information chatbot")
parser.add_argument(
    "--debug",
    action="store_true",
    help="Print the model's raw JSON response for each prompt.",
)
args = parser.parse_args()

disable_progress_bars()
transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()

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

The response must contain exactly this key:
{
  "requests": [{"smiles": "SMILES", "property": "approved_property_name"}]
}

Each item in requests must contain exactly one SMILES and one property.
If several properties are requested for one SMILES, create a separate item
for each property. The same SMILES may therefore appear in multiple items,
but an identical SMILES-property pair must not appear more than once.

For a supported property, the property value must be exactly one of these
approved names:
- molecular_weight
- logP
- TPSA
- hydrogen_bond_donors
- hydrogen_bond_acceptors
- rotatable_bonds
- rings
- aromatic_rings

Map the user's wording to one approved name. For example, molecular mass
means molecular_weight; lipophilicity and hydrophobicity mean logP; topological
polar surface area means TPSA; HBD and HBA mean hydrogen_bond_donors and
hydrogen_bond_acceptors. Never return a synonym as the property value.

For a property or task that is not supported, put a short phrase naming that
requested property/task in the property field. Do not silently omit it or
replace it with a supported property; Python will reject it and explain the
supported scope.

Recognize common spelling errors. Preserve SMILES text, including lowercase
aromatic atoms. Ignore conversational punctuation around a SMILES.

Associate each property only with the SMILES specified by the same phrase or
clause. Resolve each property-to-SMILES association separately. Do not copy a
property to another SMILES just because both occur in the same prompt. When the
user explicitly requests the same property for multiple molecules in a
comparison, create one item for each molecule. Do not infer extra properties.

Example:
User: "For CCO, give the molecular weight and HBD. For c1ccccc1, give the
logP and number of rings."
Output:
{
  "requests": [
    {"smiles": "CCO", "property": "molecular_weight"},
    {"smiles": "CCO", "property": "hydrogen_bond_donors"},
    {"smiles": "c1ccccc1", "property": "logP"},
    {"smiles": "c1ccccc1", "property": "rings"}
  ]
}

If the user asks about a supported property but supplies no SMILES, include a
request with an empty string for smiles and the approved property name. Do not
invent a SMILES from a chemical name, formula, or context. If the user asks
about an unsupported property or task, include a request with that property's
phrase in property, preserving the supplied SMILES or using an empty string if
none was supplied. If a requested property has an invalid-looking SMILES,
preserve it exactly; Python will validate it with RDKit. Return an empty
requests list only if there is no request to process.

Examples:
User: "What is the boiling point of CCO?"
Output: {"requests": [{"smiles": "CCO", "property": "boiling point"}]}

User: "What is the molecular weight?"
Output: {"requests": [{"smiles": "", "property": "molecular_weight"}]}

User: "Tell me a joke."
Output: {"requests": [{"smiles": "", "property": "joke"}]}
"""

# ---------------------------------------------------------------------------
# Load the tokenizer and model
# ---------------------------------------------------------------------------
# Tokenizer: converts text <-> token IDs the model understands.
try:
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
except (OSError, RuntimeError, ValueError, TypeError):
    logging.exception("Unable to load the tokenizer or model.")
    raise


def generate_response(messages, max_new_tokens):
    """Generate a deterministic response for the supplied chat messages."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    model.eval()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    return tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
# Keep asking the user for prompts until they type "quit" or "exit"
# (or press Ctrl+C). Each turn is independent — the model does NOT remember
# previous questions in this loop, since we rebuild `messages` from scratch
# every iteration.
while True:
    try:
        prompt = input("\nPrompt (type 'quit' to exit): ")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting the chatbot.")
        break

    if prompt.strip().lower() in {"quit", "exit", "q"}:
        break

    try:
        # 1) Build the chat as a list of role/content dicts. The system message
        #    comes first so the model reads it before the user's question.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = generate_response(messages, max_new_tokens=512)
        if args.debug:
            print(response)

        validation_error_message = agent_functions.validate_model_response(response)

        if validation_error_message and validation_error_message.startswith("Error:"):
            retry_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not pass validation: "
                        f"{validation_error_message} Re-read the original request "
                        "and return corrected JSON only, following the system schema. "
                        "Do not guess missing information or add unrequested properties."
                    ),
                },
            ]
            response = generate_response(retry_messages, max_new_tokens=512)
            if args.debug:
                print(response)
            validation_error_message = agent_functions.validate_model_response(response)

        if validation_error_message:
            print(validation_error_message)
            continue

        requests = json.loads(response)["requests"]
        output_properties = agent_functions.process_requests(requests)
        agent_functions.print_properties_tables(output_properties)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        logging.exception("Failed to process the user's prompt.")
        print(f"Sorry, I couldn't process that request: {error}")
        continue

    # -----------------------------------------------------------------------
    # Do your work to do the tool calls based on LLM response
    # -----------------------------------------------------------------------
    
    # TODO 2: Based on the output of the LLM, make the appropriate tool
    # calls to RDKit and output the response in a user friendly manner.
    # Have fun in this part of the code!
