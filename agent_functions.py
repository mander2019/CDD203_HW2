import json
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

JSON_SCHEMA = {"requests": list}

PROPERTIES = (
    "molecular_weight",
    "logP",
    "TPSA",
    "hydrogen_bond_donors",
    "hydrogen_bond_acceptors",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
)


# JSON validation for model response

def json_validation(response):
    """Validate the JSON response from the model."""

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        return "Error: The model's response is not valid JSON."

    if not isinstance(response_json, dict):
        return "Error: The model's response JSON must be an object."

    if set(response_json) != set(JSON_SCHEMA):
        return "Error: The model's response JSON does not have the expected keys."

    requests = response_json["requests"]
    if not isinstance(requests, list):
        return "Error: The 'requests' field must be a list."

    if not requests:
        return "I can only answer questions about the supported chemical properties."

    invalid_smiles = []
    allowed_properties = set(PROPERTIES)

    for request in requests:
        if not isinstance(request, dict) or set(request) != {"smiles", "property"}:
            return "Error: Each request must contain exactly 'smiles' and 'property'."

        smiles_string = request["smiles"]
        property_name = request["property"]

        if not isinstance(property_name, str):
            return "Error: Property names must be strings."

        if property_name not in allowed_properties:
            return (
                "I can only answer questions about these supported properties: "
                f"{', '.join(PROPERTIES)}."
            )

        if not isinstance(smiles_string, str) or not smiles_string.strip():
            return "Please provide a valid SMILES string."

        if Chem.MolFromSmiles(smiles_string) is None:
            invalid_smiles.append(smiles_string)

    if invalid_smiles:
        return (f"'{', '.join(invalid_smiles)}' is not a valid SMILES string. "
                "Please provide a valid SMILES string.")

    return None


def validate_model_response(response):
    """Run JSON validation and convert expected validation exceptions to errors."""
    try:
        return json_validation(response)
    except (TypeError, ValueError):
        logging.exception("Model response validation raised an exception.")
        return "Error: The model's response could not be validated."

# Text formatting functions

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

def print_properties_tables(all_output_properties):
    """Print aligned tables for multiple SMILES strings."""
    for smiles, output_properties in all_output_properties:
        print_properties_table(smiles, output_properties)
        print()  # Add a blank line between tables

# Iterate through requests

def process_requests(requests, method="rdkit"):
    """Process the requests and return a list of calculated properties."""

    properties_by_smiles = {}
    seen_requests = set()
    for request in requests:
        smiles_string = request["smiles"]
        property_name = request["property"]
        if property_name not in PROPERTIES:
            raise ValueError(f"Unsupported chemical property: {property_name}")
        request_pair = (smiles_string, property_name)
        if request_pair in seen_requests:
            continue
        seen_requests.add(request_pair)
        properties_by_smiles.setdefault(smiles_string, []).append(property_name)

    return [
        (
            smiles_string,
            get_chemical_properties(smiles_string, properties, method=method),
        )
        for smiles_string, properties in properties_by_smiles.items()
    ]

def get_chemical_properties(smiles, request, method):
    """Calculate chemical properties using RDKit."""
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    output_properties = []
    for property_name in request:
        if method == "rdkit":
            value = get_property_rdkit(smiles, property_name)
        elif method == "mcp":
            # TODO not yet implemented
            value = get_property_mcp(smiles, property_name)

        output_properties.append({
            "property": property_name,
            "value": value,
        })

    return output_properties

def get_property_rdkit(smiles, property_name):
    """Calculate a single chemical property using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if property_name == "molecular_weight":
        return Descriptors.MolWt(mol)
    elif property_name == "logP":
        return Descriptors.MolLogP(mol)
    elif property_name == "TPSA":
        return Descriptors.TPSA(mol)
    elif property_name == "hydrogen_bond_donors":
        return Descriptors.NumHDonors(mol)
    elif property_name == "hydrogen_bond_acceptors":
        return Descriptors.NumHAcceptors(mol)
    elif property_name == "rotatable_bonds":
        return Descriptors.NumRotatableBonds(mol)
    elif property_name == "rings":
        return Descriptors.RingCount(mol)
    elif property_name == "aromatic_rings":
        return Descriptors.NumAromaticRings(mol)
    else:
        return None

def get_property_mcp(smiles, property_name):
    """Calculate a single chemical property using MCP."""
    # TODO
    # MCP
    return None
