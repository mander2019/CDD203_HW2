import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors

JSON_SCHEMA = {
        "requests": list,
        "missing_smiles": bool,
        "unknown_request": bool,
    }

PROPERTIES = ["molecular_weight",
              "logP",
              "TPSA",
              "hydrogen_bond_donors",
              "hydrogen_bond_acceptors",
              "rotatable_bonds",
              "rings",
              "aromatic_rings"]

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

    if not all(
        isinstance(response_json[key], expected_type)
        for key, expected_type in JSON_SCHEMA.items()
    ):
        return "Error: The model's response JSON has incorrect data types."

    requests = response_json["requests"]
    missing_smiles = response_json["missing_smiles"]
    unknown_request = response_json["unknown_request"]

    if missing_smiles and unknown_request:
        return "Error: The model's response JSON has conflicting request flags."

    if missing_smiles or unknown_request:
        if requests:
            return "Error: The model's response JSON must have an empty requests list when a request flag is true."

    if unknown_request:
        return "I only answer questions about molecular weight, logP, TPSA, hydrogen bond donors, hydrogen bond acceptors, rotatable bonds, rings, and aromatic rings"

    if missing_smiles:
        return "Please provide a valid SMILES string"

    if not requests:
        return "Error: The model's response JSON must contain at least one request."

    invalid_smiles = []
    seen_smiles = set()
    allowed_properties = set(PROPERTIES)

    for request in requests:
        if not isinstance(request, dict) or set(request) != {"smiles", "properties"}:
            return "Error: Each request must contain exactly 'smiles' and 'properties'."

        smiles_string = request["smiles"]
        properties = request["properties"]

        if not isinstance(smiles_string, str) or not smiles_string.strip():
            return "Error: Each request must contain a non-empty SMILES string."

        if not isinstance(properties, list) or not properties:
            return "Error: Each request must contain a non-empty properties list."

        if not all(isinstance(property_name, str) for property_name in properties):
            return "Error: Property names must be strings."

        if any(property_name not in allowed_properties for property_name in properties):
            return "Error: The model requested an unsupported chemical property."

        if len(properties) != len(set(properties)):
            return "Error: A request cannot contain duplicate properties."

        if smiles_string in seen_smiles:
            return "Error: Each SMILES string must appear in only one request."
        seen_smiles.add(smiles_string)

        if Chem.MolFromSmiles(smiles_string) is None:
            invalid_smiles.append(smiles_string)

    if invalid_smiles:
        return (f"'{', '.join(invalid_smiles)}' is not a valid SMILES string. "
                "Please provide a valid SMILES string.")

    return None

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
    
    all_output_properties = []

    for request in requests:
        properties = request.get("properties", [])
        smiles_string = request["smiles"]

        smiles_properties = get_chemical_properties(smiles_string, properties, method=method)
        all_output_properties.append((smiles_string, smiles_properties))
    
    return all_output_properties

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
