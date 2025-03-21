"""Plot functions for visualization of mesh and results."""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import deflation
import mesher



def plotMesh(mesh: mesher.Mesher,
             bc = None,
             u = None,
             uLimits = None,
             cmap='jet',
             show_edges=True, 
	           window_size=(716, 538),
             background_color='white',
             edge_color='black',
	           title='Mesh Visualization',
             save_path=None,
             fontsize=10):
  # Create vertices array
  vertices = mesh.node_xyz
  
  # Handle deformation if provided
  if (u is not None) and mesh.dofs_per_node == 3 and (np.max(np.abs(u))> 0):  # structural
    delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    deltaMax = np.max(delta)
    scale = 0.1*mesh.bbox.diag_length/deltaMax
    uVertex = u.reshape(vertices.shape)
    vertices = vertices + scale*uVertex
    values = delta
  elif (u is not None) and mesh.dofs_per_node == 1 and (np.max(np.abs(u))> 0):  # thermal
    values = u
  else:
    values = None

  # Match plotMeshOld exactly
  faceIndex = np.array([[0,4,7,3],
                        [0,1,5,4],
                        [0,3,2,1],
                        [1,2,6,5],
                        [2,3,7,6],
                        [4,5,6,7]], dtype=np.uint32)
  nFacesPerHex = 6
  faces = []
  face_densities = []
  
  for e in range(mesh.num_elems):
    if mesh.elemPseudoDensity[e] < 0.5:
      continue
    elif (mesh.elemPseudoDensity[e] > 0.5 and 
          np.all(mesh.elemNeighborsArray[e] > 0) and 
          np.all(mesh.elemPseudoDensity[[int(elem) for elem in 
                                    mesh.elemNeighborsArray[e]]] > 0.5)):
      continue

    # Add all faces for this element
    for j in range(nFacesPerHex):
      faces.append(mesh.elemArray[e,faceIndex[j,:]])
      face_densities.append(mesh.elemPseudoDensity[e])

  # Convert to numpy arrays
  faces = np.array(faces)
  face_densities = np.array(face_densities)
  
  if len(faces) == 0:
    raise ValueError("No faces to plot after filtering")

  # Create cells array for PyVista
  n_faces = len(faces)
  cells = np.hstack((
                    np.full((n_faces, 1), 4),  # 4 vertices per face
                    faces
                    ))

  # Create PyVista mesh
  pv_mesh = pv.UnstructuredGrid({9: cells[:, 1:]}, vertices)  # 9 is VTK_QUAD

  # Add scalar values
  if values is not None:
    pv_mesh.point_data['values'] = values
  
  # Add density values to cells
  pv_mesh.cell_data['density'] = face_densities

  # Create plotter
  if save_path is  None:
    plotter = pv.Plotter(window_size=window_size)
  else:
    plotter = pv.Plotter(window_size=window_size,off_screen=True)
  plotter.set_background(background_color)

  # Add mesh to plotter
  if u is not None:
    if (uLimits is not None):
      plotter.add_mesh(
              pv_mesh,
              scalars='values' if values is not None else 'density',
              cmap=cmap,
              show_edges=show_edges,
              edge_color=edge_color,
              line_width=1,
              clim=uLimits,  # Set color limits
              scalar_bar_args={ 
                      'title': '',
                      'vertical': True,
                      'position_x': 0.8,
                      'position_y': 0.3,
                      'width': 0.1
                      }
            )
    else:
        plotter.add_mesh(
                  pv_mesh,
                  scalars='values' if values is not None else 'density',
                  cmap=cmap,
                  show_edges=show_edges,
                  edge_color=edge_color,
                  line_width=1,
                  scalar_bar_args={
                          'title': '',
                          'vertical': True,
                          'position_x': 0.8,
                          'position_y': 0.3,
                          'width': 0.1
                          }
                )
  else:
    # Light green color when no deformation
    plotter.add_mesh(
                    pv_mesh,
                    color='lightgreen',
                    show_edges=show_edges,
                    edge_color=edge_color,
                    line_width=1
                  )

  # Add dots and force arrows for labeled nodes
  point_size = 10  # Size of dots in pixels

  # Add black dots for label 1 (fixed nodes)
  label1_nodes = np.where(mesh.node_indices[:, 3] == 1)[0]
  if len(label1_nodes) > 0 and bc is not None:
    points1 = vertices[label1_nodes]
    dots1 = pv.PolyData(points1)
    plotter.add_points(dots1,
                       color='black',
                       point_size=point_size,
                       render_points_as_spheres=True)

  # Add force arrows for label 2 (without red dots)
  label2_nodes = np.where(mesh.node_indices[:, 3] == 2)[0]
  if len(label2_nodes) > 0 and mesh.dofs_per_node == 3 and bc is not None: #structural
    # Add force arrows
    arrow_scale = 0.1 * mesh.bbox.diag_length
    for node in label2_nodes:
      # Get force components for this node
      fx = bc.force[3*node]
      fy = bc.force[3*node + 1]
      fz = bc.force[3*node + 2]
      force_vec = np.array([fx, fy, fz])
      
      # Only add arrow if force is non-zero
      if np.linalg.norm(force_vec) > 0:
        # Normalize and scale force vector
        force_vec = force_vec / np.linalg.norm(force_vec) * arrow_scale
        
        # Create arrow
        start_point = vertices[node]
  
        # Add arrow to plot
        arrow = pv.Arrow(start = start_point,
                         direction = force_vec,
                         scale = arrow_scale)
        plotter.add_mesh(arrow, color='red')
  if len(label2_nodes) > 0 and mesh.dofs_per_node == 1 and bc is not None: #thermal
    # Add force arrows
    
    for node in label2_nodes:
      # Get force components for this node
      q = bc.force[node]

      # Only add arrow if flux is non-zero
      if np.abs(q) > 0:
        point = vertices[node]
        plotter.add_points(point,
                       color='red',
                       point_size=point_size,
                       render_points_as_spheres=True)
  # Add title
  if title:
    plotter.add_title(title, font_size=fontsize)

  # Add coordinate axes widget
  plotter.add_axes(
                  xlabel='X',
                  ylabel='Y',
                  zlabel='Z',
                  line_width=2,
                  labels_off=False,  # Show axis labels
                  color='black'
                  )

  # Set camera position for left-bottom-forward view
  view_distance = 2.5 * mesh.bbox.diag_length
  offset = 0.2 * view_distance  # Offset for object position
  plotter.camera_position = [
                  (view_distance*0.5, -view_distance*0.3, view_distance),
                  (offset, offset, 0),   # Focus point - right and bottom
                  (0, 0.8, 0.4)]         # Up vector - Y axis up

  # Reset camera and zoom out slightly
  plotter.camera.zoom(0.8)
  
  # Enable anti-aliasing for better quality
  plotter.enable_anti_aliasing()
  
  # Save image if path is provided
  if save_path:
    #plotter.show(screenshot = save_path)
    plotter.screenshot(save_path)
    plotter.close()
  else:
    plotter.show() 
  
  return plotter

def plotElementField(mesh: mesher.Mesher,
            field,
            cmap='jet',
            show_edges=True,
            window_size=(716, 538),
            background_color='white',
            edge_color='black',
            title='Element Field Visualization',
            save_path=None,
            fontsize=10):
    """Plot element field on the mesh.

    Args:
    mesh (mesher.Mesher): The mesh object.
    field (ndarray): Element field values.
    cmap (str): Colormap for visualization.
    show_edges (bool): Whether to show mesh edges.
    window_size (tuple): Window size for visualization.
    background_color (str): Background color.
    edge_color (str): Edge color.
    title (str): Plot title.
    save_path (str, optional): Path to save the visualization.
    fontsize (int): Font size for title.
    """
    # Create vertices array
    vertices = mesh.node_xyz

    # Create cells array for PyVista
    cells = np.hstack((
              np.full((mesh.num_elems, 1), 8),  # 8 vertices per hexahedron
              mesh.elemArray
            ))

    # Create PyVista mesh
    pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

    # Add field data to cell data
    pv_mesh.cell_data['field'] = field

    # Create plotter
    if save_path is None:
      plotter = pv.Plotter(window_size=window_size)
    else:
      plotter = pv.Plotter(window_size=window_size, off_screen=True)
      plotter.set_background(background_color)

    # Add mesh to plotter
    plotter.add_mesh(
            pv_mesh,
            scalars='field',
            cmap=cmap,
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=1,
            scalar_bar_args={
              'title': '',
              'vertical': True,
              'position_x': 0.8,
              'position_y': 0.3,
              'width': 0.1
            }
          )

    # Add title
    if title:
      plotter.add_title(title, font_size=fontsize)

    # Add coordinate axes widget
    plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False,  # Show axis labels
            color='black'
            )

    # Set camera position for left-bottom-forward view
    view_distance = 2.5 * mesh.bbox.diag_length
    offset = 0.2 * view_distance  # Offset for object position
    plotter.camera_position = [
            (view_distance*0.5, -view_distance*0.3, view_distance),
            (offset, offset, 0),   # Focus point - right and bottom
            (0, 0.8, 0.4)]         # Up vector - Y axis up

    # Reset camera and zoom out slightly
    plotter.camera.zoom(0.8)

    # Enable anti-aliasing for better quality
    plotter.enable_anti_aliasing()

    # Save image if path is provided
    if save_path:
      plotter.screenshot(save_path)
      plotter.close()
    else:
      plotter.show()

    return plotter

def plotIsocontour(mesh: mesher.Mesher,
                   u=None,
                   Binarization = False,
                   isovalue=0.5,
                   show_edges=True,
                   resolution=1,
                   window_size=(716, 538),
                   background_color='white',
                   edge_color='black',
                   title='Isocontour Visualization',
                   save_path=None,
                   interactive=False,
                   fontsize=10):
  """Plot isocontour surface based on element pseudo-density.

  Args:
    u (ndarray, optional): Displacement vector for deformation visualization
    isovalue (float): Isocontour value (default: 0.5)
    cmap (str): Colormap for visualization
    show_edges (bool): Whether to show mesh edges
    window_size (tuple): Window size for visualization
    background_color (str): Background color
    edge_color (str): Edge color
    title (str): Plot title
    save_path (str, optional): Path to save the visualization
    fontsize (int): Font size for title
  """
  # Create vertices array
  vertices = mesh.node_xyz
  
  # Handle deformation if provided
  if (u is not None) and (np.max(np.abs(u)) > 0):
    delta = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    deltaMax = np.max(delta)
    scale = 0.1*mesh.bbox.diag_length/deltaMax
    uVertex = u.reshape(vertices.shape)
    vertices = vertices + scale*uVertex
    values = delta
  else:
    values = None

  # Create cells array for PyVista
  cells = np.hstack((
                      np.full((mesh.num_elems, 1), 8),  # 8 vertices per hexahedron
                      mesh.elemArray
                    ))

  # Create PyVista mesh
  pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

  elemPseudoDensity = mesh.elemPseudoDensity
  if (Binarization):
    elemPseudoDensity = np.where(elemPseudoDensity > 0.5, 1, 0)
  # Add element densities to cell data
  pv_mesh.cell_data['density'] = elemPseudoDensity

  # Add displacement values to point data if provided
  if values is not None:
    pv_mesh.point_data['displacement'] = values

  # Convert cell data to point data
  mesh_with_point_data = pv_mesh.cell_data_to_point_data()

  # Create a larger grid that encompasses the mesh with buffer
  bounds = pv_mesh.bounds
  padding = max([bounds[1]-bounds[0],
                 bounds[3]-bounds[2],
                 bounds[5]-bounds[4]]) * 0.1

  # Optionally, create a finer grid for better isosurface
  
  dimensions = (resolution*mesh.grid[0], resolution*mesh.grid[1], resolution*mesh.grid[2])
  if (min(dimensions) < 2):
    resolution = 3
  spacing = (
            (bounds[1] - bounds[0] + 2*padding) / (dimensions[0] - 1),
            (bounds[3] - bounds[2] + 2*padding) / (dimensions[1] - 1),
            (bounds[5] - bounds[4] + 2*padding) / (dimensions[2] - 1)
            )
  origin = (bounds[0] - padding, bounds[2] - padding, bounds[4] - padding)

  # Create grid
  grid = pv.ImageData(
                      dimensions=dimensions,
                      spacing=spacing,
                      origin=origin
                    )
  
  # Interpolate data onto grid
  grid_with_data = grid.interpolate(mesh_with_point_data, radius=padding/2, null_value=-10)

  # Set values outside the mesh to a large -ve value
  # to ensure they are not included in the isocontour
  grid_mask = (~grid_with_data.point_data['density'].mask 
                if hasattr(grid_with_data.point_data['density'], 'mask')
                else None)
  if grid_mask is not None:
    grid_with_data.point_data['density'][grid_mask] = 0
  
  # Generate isocontour
  isosurf = grid_with_data.contour([isovalue], scalars='density')

  plotter = pv.Plotter(window_size=window_size)
  plotter.set_background(background_color)
  
  # Add isocontour to plotter
  if u is not None:
    plotter.add_mesh(
                    isosurf,
                    color='lightgreen',
                    show_edges=show_edges,
                    edge_color=edge_color,
                    line_width=1
                    )
  else:
    plotter.add_mesh(
                    isosurf,
                    color='lightgreen',
                    show_edges=show_edges,
                    edge_color=edge_color,
                    line_width=1
                   )
  
  # Add title
  if title:
    plotter.add_title(title, font_size=fontsize)

  # Add coordinate axes widget
  plotter.add_axes(
                  xlabel='X',
                  ylabel='Y',
                  zlabel='Z',
                  line_width=2,
                  labels_off=False,
                  color='black'
                  )

  # Set camera position
  view_distance = 2.5 * mesh.bbox.diag_length
  offset = 0.2 * view_distance
  plotter.camera_position = [
                        (view_distance*0.5, -view_distance*0.3, view_distance),
                        (offset, offset, 0),
                        (0, 0.8, 0.4)]

  # Reset camera and zoom out slightly
  plotter.camera.zoom(0.8)
  
  # Enable anti-aliasing
  plotter.enable_anti_aliasing()
  
  # Save or show
  if save_path:
    plotter.show(save_path  = save_path)
  else:
    plotter.show()
  
  return plotter


def plot_deflation_groups(deflation: deflation.DeflationSolver,
                          mesh: mesher.Mesher):
  # Create vertices array
  vertices = np.zeros((mesh.num_nodes,3))
  vertices = mesh.origin + mesh.elem_size * mesh.node_indices[:,:3]

  # Create PyVista point cloud
  points = pv.PolyData(vertices)

  # Add group numbers as scalar data
  points.point_data['groups'] = deflation.ws_nodeGroupNumber

  # Create plotter
  plotter = pv.Plotter()
  plotter.set_background('white')

  # Add points with group coloring
  plotter.add_mesh(points, scalars='groups', 
                  point_size=7,
                  render_points_as_spheres=True,
                  cmap='rainbow')

  # Add axes
  plotter.add_axes()

  # Set camera for isometric view
  plotter.view_isometric()
  plotter.show()


def plotGroupCenter(deflation: deflation.DeflationSolver):
  fig = plt.figure()
  ax = fig.add_subplot(projection = '3d')
  ax.scatter(deflation.ws_groupCenter[:, 0],
             deflation.ws_groupCenter[:, 1],
             deflation.ws_groupCenter[:, 2],
             '*')