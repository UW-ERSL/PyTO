"""Handler for material data."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class StructuralMaterial:
  """Linear structural material constants.

  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
  """
  youngs_modulus: Optional[float] = 2.1e11  # Pa
  poissons_ratio: Optional[float] = 0.28  # [-]
  mass_density: Optional[float] = 7700.0  # kg/m^3

@dataclasses.dataclass
class ThermalMaterial:
  """Linear thermal material constants.

  Attributes:
    thermal_conductivity: The thermal conductivity of the material [W/mK].
    specific_heat: The specific heat of the material [J/kgK].
  """
  thermal_conductivity: Optional[float] = None
  specific_heat: Optional[float] = None
  mass_density: Optional[float] = None
