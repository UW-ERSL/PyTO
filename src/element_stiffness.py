"""Stiffness matrix template for a 3D hexahedral element."""

import numpy as np
import scipy.special as spy_spl

import mat_lib


def get_gauss_integ_points_weights(order: int,
                                   dimension: int,
                                   )->tuple[np.ndarray, np.ndarray]:
  """
  Returns the Gauss integration points and weights for the given order and
    dimension. The number of gauss points is order^dimension.

  Args:
    order (int): The order of the Gauss quadrature.
    dimension (int): The dimension of the integration. Must be in range (1, 3).

  Returns:
    tuple: A tuple containing the integration points and weights.
    - points (numpy.ndarray): An array of shape (num_gauss_pts, dimension)
       containing the integration points.
    - weights (numpy.ndarray): An array of shape (num_gauss_pts,) containing the
        integration weights.

  Raises:
      ValueError: If dimension is not in (1, 3).
  """
  # Get 1D Gauss points and weights
  x, w = spy_spl.roots_legendre(order)

  if dimension == 1:
    points = x.reshape(-1, 1)
    weights = w
  elif dimension == 2:
    # Generate 2D points and weights as tensor products of 1D points and weights
    points = np.array([[x_i, x_j] for x_i in x for x_j in x])
    weights = np.array([w_i * w_j for w_i in w for w_j in w])
  elif dimension == 3:
    # Generate 3D points and weights as tensor products of 1D points and weights
    points = np.array([[x_i, x_j, x_k] for x_i in x for x_j in x for x_k in x])
    weights = np.array([w_i * w_j * w_k for w_i in w for w_j in w for w_k in w])
  else:
    raise ValueError("Dimension must be in (1, 3)")

  return points, weights


class Hex8Element:
  """Hexahedral element with 8 nodes.

  The nodes are numbered as follows:
                             7-------6
                            /|      /|
    z   y                  4-------5 |
    | /                    | 3-----| 2
    |/                     |/      |/
    0--------x             0-------1
  The edges are numbered as follows: (bottom to top, front to back, acyclic)
    0: 0-1, 1: 1-2, 2: 2-3, 3: 3-0, (bottom face)
    4: 4-5, 5: 5-6, 6: 6-7, 7: 7-4, (top face)
    8: 0-4, 9: 1-5, 10: 2-6, 11: 3-7 (vertical edges front to back, bottom to top)
  """


  @property
  def dimension(self)->int:
    return 3


  @property
  def num_nodes(self)->int:
    return 8


  @staticmethod
  def edge_connectivity()->tuple[np.ndarray, np.ndarray]:
    """Connectivity of the edges of the element.

    Returns: A tuple containing the connectivity of the edges of the element.
      The first array contains the start node of the edge and the second array
      contains the end node of the edge.
    """
    return (np.array([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]).astype(int),
            np.array([1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]).astype(int))


  @staticmethod
  def shape_functions(gauss_pts: np.ndarray,
                      )->np.ndarray:
    """Compute the shape functions at the given xi, eta, zeta.

    Args:
      gauss_pts: Array of (g, 3) containing the isoparametric xi, eta, zeta
        coordinates. The `g` is usually the number of gauss points.

    Returns: Array of shape (g, 8) containing the shape functions at the given 
      xi, eta, zeta. The 8 corresponds to the number of nodes in the element.
    """
    xi, eta, zeta = gauss_pts[:, 0], gauss_pts[:, 1], gauss_pts[:, 2]
    xis = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    etas = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zetas = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    return np.array([
                      0.125 * (1 + xi * i) * (1 + eta * j) * (1 + zeta * k)
                      for i, j, k in zip(xis, etas, zetas)
                    ]).T


  @staticmethod
  def shape_function_gradients_isoparametric(
                              gauss_pts: np.ndarray,
                              )->np.ndarray:
    """Compute the shape function gradients at the given xi, eta, zeta.

    Args:
      gauss_pts: Array of (g, 3) containing the isoparametric xi, eta, zeta
        coordinates. The `g` is usually the number of gauss points.

    Returns: Array of (g, 8, 3) arrays containing the shape function gradients at
      given xi, eta, zeta. The 8 corresponds to the number of nodes in the element
      and 3 corresponds to the number of physical dimensions.
    """
    xi, eta, zeta = gauss_pts[:, 0], gauss_pts[:, 1], gauss_pts[:, 2]
    xis = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    etas = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    zetas = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    dN_dxi = np.array([
                        0.125 * i * (1 + eta * j) * (1 + zeta * k)
                        for i, j, k in zip(xis, etas, zetas)
                      ]).T
    dN_deta = np.array([
                        0.125 * (1 + xi * i) * j * (1 + zeta * k)
                        for i, j, k in zip(xis, etas, zetas)
                      ]).T
    dN_dzeta = np.array([
                        0.125 * (1 + xi * i) * (1 + eta * j) * k
                        for i, j, k in zip(xis, etas, zetas)
                      ]).T              
    return np.stack((dN_dxi, dN_deta, dN_dzeta), axis=2)


  @classmethod
  def compute_jacobian_and_determinant(cls,
                                    gauss_pts: np.ndarray,
                                    xyz_nodes: np.ndarray,
                                    )-> tuple[np.ndarray, np.ndarray]:
    """Compute the Jacobian of the element at the given xi, eta, zeta.

    Args:
      gauss_pts: Array of (g, 3) containing the isoparametric xi, eta, zeta
        coordinates. The `g` is usually the number of gauss points.
      xyz_nodes: Array of (8, 3) containing the x, y, z coordinates of the nodes.

    Returns: A tuple containing the Jacobian and the its determinant:
      - Jacobians of shape (g, 3, 3) containing the Jacobian of the element at
          the given xi, eta, zeta. 3 corresponds to the number of physical dims.
      - Determinant of shape (g,) containing the determinant of the Jacobian.
    """
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pts)
    # (g)auss points, num_(n)odes, (d)[i]m
    jac = np.einsum('gnd, ni -> gdi', gradN_isoparam, xyz_nodes)
    det_jac  = np.linalg.det(jac)
    return jac, det_jac


  def get_gradient_shape_function_physical(cls,
                                          gauss_pts: np.ndarray,
                                          xyz_nodes: np.ndarray,
                                          )->np.ndarray:
    """Compute the gradient of the shape functions at the given gauss point.

    Args:
      gauss_pts: Array of (g, 3) containing the isoparametric xi, eta and zeta
        coordinates. The `g` is usually the number of gauss points.
      xyz_nodes: Array of (8, 3) containing the x, y and z coordinates of the nodes.

    Returns: An array of shape (g, 8, 3) containing the derivatives of the shape
      functions with respect to the physical coordinates. The 8 corresponds to the
      number of nodes in the element and 3 corresponds to the number of physical
      dimensions.
    """
    # (g)auss points, num_(n)odes, (d)[i]m
    gradN_isoparam = cls.shape_function_gradients_isoparametric(gauss_pts) # {gnd}
    jac, _ = cls.compute_jacobian_and_determinant(gauss_pts, xyz_nodes)
    return np.einsum('gdi, gnd->gni', np.linalg.inv(jac), gradN_isoparam)


_isotropic_constitutive_matrix = lambda e, nu: (
                              (e/((1. + nu) * (1. - 2. * nu))) * 
                              np.array([[1. - nu, nu, nu, 0., 0., 0.],
                                        [nu, 1. - nu, nu, 0., 0., 0.],
                                        [nu, nu, 1. - nu, 0., 0., 0.],
                                        [0., 0., 0., (1. - 2. * nu)/2., 0., 0.],
                                        [0., 0., 0., 0., (1. - 2. * nu)/2., 0.],
                                        [0., 0., 0., 0., 0., (1. - 2. * nu)/2.]
                                        ])
                                  )



def hex8_stiffness_matrix_structural(
              mat_prop: mat_lib.StructuralMaterial,
              elem_size: tuple[float, float, float],
              gauss_order: int = 6,
            ) -> np.ndarray:
  """Computes the element stiffness matrix of a hexahedral element in 3D.
    The stiffness matrix for linear structural elasticity is derived as:

                K = sum_gauss(B'D B |J| w )
      Where,

          B is the strain displacement matrix.
          D is the constitutive matrix.
          J is the Jacobian.
          w is the gauss weight.
  Args:
    mat_prop: The structual material properties of the element.
    elem_size: The size of the element in x, y, z directions.
    gauss_order: The Gauss integration order.

  Returns: The element stiffness matrix of size (24, 24). The 24 corresponds to
    the 8 nodes and 3 DOF per node.
  """
  dx, dy, dz = elem_size
  nodes = np.array([[0, dx, dx, 0, 0, dx, dx, 0],
                    [0, 0, dy, dy, 0, 0, dy, dy],
                    [0, 0, 0, 0, dz, dz, dz, dz]]).T
  c = _isotropic_constitutive_matrix(mat_prop.youngs_modulus,
                                     mat_prop.poissons_ratio)

  # Initialize the element stiffness matrix.
  ke = np.zeros((24, 24))

  elem = Hex8Element()
  gauss_pts, gauss_wts = get_gauss_integ_points_weights(gauss_order,
														                            elem.dimension)

  jacobians, _ = elem.compute_jacobian_and_determinant(gauss_pts, nodes)
  dN_dxyz = elem.get_gradient_shape_function_physical(gauss_pts, nodes)

  for ctr in range(gauss_wts.shape[0]):
    jac = jacobians[ctr, :, :]
    dn_dx, dn_dy, dn_dz = dN_dxyz[ctr, :, 0], dN_dxyz[ctr, :, 1], dN_dxyz[ctr, :, 2]
    # Compute the strain-displacement matrix.
    b = np.zeros((6, 24))
    for l in range(8):
      b[0, 3 * l] = dn_dx[l]
      b[1, 3 * l + 1] = dn_dy[l]
      b[2, 3 * l + 2] = dn_dz[l]
      b[3, 3 * l] = dn_dy[l]
      b[3, 3 * l + 1] = dn_dx[l]
      b[4, 3 * l + 1] = dn_dz[l]
      b[4, 3 * l + 2] = dn_dy[l]
      b[5, 3 * l] = dn_dz[l]
      b[5, 3 * l + 2] = dn_dx[l]
 
    ke += b.T @ c @ b * np.linalg.det(jac) *gauss_wts[ctr]
  return ke

def hex8_mass_matrix_structural(
        mat_prop: mat_lib.StructuralMaterial, 
        elem_size: tuple[float, float, float],
        gauss_order: int = 6,
      ) -> np.ndarray:
  """Computes the element mass matrix of a hexahedral element in 3D.
  The mass matrix for structural dynamics is derived as:

        M = sum_gauss(rho * N'N |J| w )
    Where,
      rho is the density
      N are the shape functions
      J is the Jacobian
      w is the gauss weight
      
  Args:
  mat_prop: The structural material properties of the element.
  elem_size: The size of the element in x, y, z directions.
  gauss_order: The Gauss integration order.

  Returns: The element mass matrix of size (24, 24). The 24 corresponds to
  the 8 nodes and 3 DOF per node.
  """
  dx, dy, dz = elem_size
  nodes = np.array([[0, dx, dx, 0, 0, dx, dx, 0],
          [0, 0, dy, dy, 0, 0, dy, dy],
          [0, 0, 0, 0, dz, dz, dz, dz]]).T

  # Initialize the element mass matrix
  me = np.zeros((24, 24))

  elem = Hex8Element()
  gauss_pts, gauss_wts = get_gauss_integ_points_weights(gauss_order,
                            elem.dimension)

  jacobians, _ = elem.compute_jacobian_and_determinant(gauss_pts, nodes)
  N = elem.shape_functions(gauss_pts)

  for ctr in range(gauss_wts.shape[0]):
    jac = jacobians[ctr, :, :]
    # Build block diagonal matrix with shape functions
    n_mat = np.zeros((3, 24))
    for i in range(8):
      n_mat[0, 3*i] = N[ctr, i]
      n_mat[1, 3*i + 1] = N[ctr, i]
      n_mat[2, 3*i + 2] = N[ctr, i]
  
    me += mat_prop.mass_density * n_mat.T @ n_mat * np.linalg.det(jac) * gauss_wts[ctr]

  return me

def hex8_stiffness_matrix_thermal(
              mat_prop: mat_lib.ThermalMaterial,
              elem_size: tuple[float, float, float],
              gauss_order: int = 6,
            ) -> np.ndarray:
  """Computes the element stiffness matrix of a hexahedral element in 3D.
    The stiffness matrix for linear thermal is derived as:

                K = sum_gauss(B'D B |J| w )
      Where,

          B is the strain displacement matrix.
          D is the constitutive matrix.
          J is the Jacobian.
          w is the gauss weight.
  Args:
    mat_prop: The thermal material properties of the element.
    elem_size: The size of the element in x, y, z directions.
    gauss_order: The Gauss integration order.

  Returns: The element stiffness matrix of size (8, 8)
  """
  dx, dy, dz = elem_size
  nodes = np.array([[0, dx, dx, 0, 0, dx, dx, 0],
                    [0, 0, dy, dy, 0, 0, dy, dy],
                    [0, 0, 0, 0, dz, dz, dz, dz]]).T

  K = mat_prop.thermal_conductivity
  # Initialize the element stiffness matrix.
  ke = np.zeros((8, 8))

  elem = Hex8Element()
  gauss_pts, gauss_wts = get_gauss_integ_points_weights(gauss_order,
														                            elem.dimension)

  jacobians, _ = elem.compute_jacobian_and_determinant(gauss_pts, nodes)
  dN_dxyz = elem.get_gradient_shape_function_physical(gauss_pts, nodes)

  for ctr in range(gauss_wts.shape[0]):
    jac = jacobians[ctr, :, :]
    dn_dx, dn_dy, dn_dz = dN_dxyz[ctr, :, 0], dN_dxyz[ctr, :, 1], dN_dxyz[ctr, :, 2]
    # Compute the strain-displacement matrix.
    b = np.zeros((3, 8))
    for l in range(8):
      b[0, l] = dn_dx[l]
      b[1, l] = dn_dy[l] 
      b[2, l] = dn_dz[l]
   
    ke += K * b.T  @ b * np.linalg.det(jac) *gauss_wts[ctr]
  return ke

def hex8_specific_heat_matrix(
        mat_prop: mat_lib.ThermalMaterial,
        elem_size: tuple[float, float, float],
        gauss_order: int = 6,
      ) -> np.ndarray:
  """Computes the element specific heat matrix for a hexahedral element in 3D.
  The specific heat matrix is derived as:

        C = sum_gauss(rho * c * N'N |J| w )
    Where,
      rho is the density
      c is the specific heat capacity
      N are the shape functions
      J is the Jacobian
      w is the gauss weight

  Args:
  mat_prop: The thermal material properties of the element
  elem_size: The size of the element in x, y, z directions
  gauss_order: The Gauss integration order

  Returns: The element specific heat matrix of size (8, 8)
  """
  dx, dy, dz = elem_size
  nodes = np.array([[0, dx, dx, 0, 0, dx, dx, 0],
          [0, 0, dy, dy, 0, 0, dy, dy],
          [0, 0, 0, 0, dz, dz, dz, dz]]).T

  # Initialize the element matrix
  ce = np.zeros((8, 8))

  elem = Hex8Element()
  gauss_pts, gauss_wts = get_gauss_integ_points_weights(gauss_order, 
                              elem.dimension)

  jacobians, _ = elem.compute_jacobian_and_determinant(gauss_pts, nodes)
  N = elem.shape_functions(gauss_pts)

  for ctr in range(gauss_wts.shape[0]):
    jac = jacobians[ctr, :, :]
    ce += (mat_prop.mass_density * mat_prop.specific_heat * 
          N[ctr, :].reshape(-1, 1) @ N[ctr, :].reshape(1, -1) * 
          np.linalg.det(jac) * gauss_wts[ctr])
  
  return ce


if __name__ == "__main__":
  # Define the material properties
  mat_prop = mat_lib.StructuralMaterial(youngs_modulus=1e6,
                                        poissons_ratio=0.3)
  print("Material Properties:")
  print(mat_prop)
  elem_size = (1, 1, 1)
  ke = hex8_stiffness_matrix_structural(mat_prop, elem_size)


  me = hex8_mass_matrix_structural(mat_prop, elem_size)
  
  # Test thermal stiffness matrix
  thermal_mat_prop = mat_lib.ThermalMaterial(thermal_conductivity=1.0)
  ke_thermal = hex8_stiffness_matrix_thermal(thermal_mat_prop, elem_size)
 