"""Handler for material data."""

import dataclasses
from typing import Optional
import difflib


@dataclasses.dataclass
class Material:
  """General material properties.

  Attributes:
    name: Name of the material.
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
    thermal_conductivity: The thermal conductivity of the material [W/mK].
    specific_heat: The specific heat of the material [J/kgK].
    thermal_expansion: Coefficient of thermal expansion [1/K].
    cost: Cost per kg in USD [$/kg].
    yield_strength: Yield strength of material [Pa].
  """
  name: str
  youngs_modulus: float
  poissons_ratio: float
  mass_density: float
  thermal_conductivity: float
  specific_heat: float
  thermal_expansion: float
  cost: float
  yield_strength: float

# Example list of materials
materials = [
  Material(
    name="Steel",
    youngs_modulus=2.1e11,
    poissons_ratio=0.3,
    mass_density=7850,
    thermal_conductivity=50,
    specific_heat=500,
    thermal_expansion=12e-6,
    cost=1.0,
    yield_strength=250e6
  ),
  Material(
    name="Aluminum",
    youngs_modulus=7.0e10,
    poissons_ratio=0.33,
    mass_density=2700,
    thermal_conductivity=237,
    specific_heat=900,
    thermal_expansion=23e-6,
    cost=2.7,
    yield_strength=95e6
  ),
  Material(
    name="Copper",
    youngs_modulus=1.1e11,
    poissons_ratio=0.34,
    mass_density=8960,
    thermal_conductivity=398,
    specific_heat=385,
    thermal_expansion=17e-6,
    cost=7.0,
    yield_strength=70e6
  ),
  Material(
    name="Titanium",
    youngs_modulus=1.16e11,
    poissons_ratio=0.34,
    mass_density=4500,
    thermal_conductivity=21.9,
    specific_heat=520,
    thermal_expansion=8.6e-6,
    cost=35.0,
    yield_strength=880e6
  ),
  Material(
    name="Nitronic60",
    youngs_modulus=1.81e11,
    poissons_ratio=0.3,
    mass_density=7620,
    thermal_conductivity=21.9,
    specific_heat=520,
    thermal_expansion=8.6e-6,
    cost=4.0,
    yield_strength=880e6
  )
]

def get_material(name: str) -> Material:
  """Get a material by name.
  
  Args:
    name: Name of the material to find.
    
  Returns:
    Material object matching the name.
    
  Raises:
    ValueError: If material name not found.
  """
  for material in materials:
    if material.name.lower() == name.lower():
      return material
  # If no exact match is found, try to find the closest one
  available_materials = [m.name.lower() for m in materials]
  closest_matches = difflib.get_close_matches(name.lower(), available_materials)
  
  if closest_matches:
    suggestion = next((m for m in materials if m.name.lower() == closest_matches[0]), None)
    raise ValueError(f"Material '{name}' not found. Did you mean '{suggestion.name}'?")
  else:
    raise ValueError(f"Material '{name}' not found")

def create_material_with_defaults(name: str, **kwargs) -> Material:
  """Create a material with some properties, defaulting others to steel values.
  
  Args:
    name: Name for the new material
    **kwargs: Material properties to override
    
  Returns:
    Material object with specified properties and steel defaults
  """
  steel = get_material("Steel")
  properties = dataclasses.asdict(steel)
  properties["name"] = name
  properties.update(kwargs)
  return Material(**properties)
