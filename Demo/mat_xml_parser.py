"""Handler for material data from xml."""
import xml.etree.ElementTree as ET
 
def convert_to_si(value_str, unit):
    """
    Converts the value string with the given unit to SI (Pa).
    Supported units: GPa, MPa, kPa, Pa.
    """
    conversions = {
        "GPa": 1e9,
        "MPa": 1e6,
        "kPa": 1e3,
        "Pa": 1,
    }
    factor = conversions.get(unit, 1)  # default to 1 if unit is unknown
    return float(value_str) * factor

def parse_material_properties(file_path):
    """
    Parses the XML file at file_path (after converting curly quotes to standard quotes)
    and returns a tuple with:
    - elastic_modulus in SI units (Pa)
    - poisons_ratio as a float
    """
    # Read and preprocess the XML file to replace curly quotes with standard quotes
    with open(file_path, "r", encoding="utf-8") as f:
        xml_content = f.read()
    xml_content = xml_content.replace("“", "\"").replace("”", "\"")
    
    root = ET.fromstring(xml_content)
    
    elastic_modulus = None
    poisons_ratio = None

    for material in root.findall("material"):
        properties = material.find("properties")
        if properties is not None:
            em_element = properties.find("property[@name='elastic_modulus']")
            pr_element = properties.find("property[@name='poisons_ratio']")
            
            if em_element is not None:
                em_value, em_unit = em_element.get("value"), em_element.get("unit")
                # Convert to SI units
                elastic_modulus = convert_to_si(em_value, em_unit)
            if pr_element is not None:
                # For poisons_ratio, we assume it's unitless and can be converted to float
                poisons_ratio = float(pr_element.get("value"))
            
            # Use the first matching material
            if elastic_modulus is not None or poisons_ratio is not None:
                break

    return elastic_modulus, poisons_ratio
