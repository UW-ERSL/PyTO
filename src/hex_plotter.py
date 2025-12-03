"""
Consolidated plotting module for HexMesh FEA visualization.

This module contains all plotting functionality for:
- Structural FEA (HexStructuralFEA)
- Thermal FEA (HexThermalFEA)
- Modal FEA (ModalFEA)
- Mesh visualization

Usage:
    from hex_fea_plotter import HexFEAPlotter
    
    plotter = HexFEAPlotter(mesh, camera_position=camera_pos)
    plotter.plot_pseudo_density(...)
    plotter.plot_structural_deformation(sol, ...)
    plotter.plot_temperature(temperature, ...)
"""

import numpy as np
import pyvista as pv
from typing import Optional, Tuple

try:
    from pyvistaqt import BackgroundPlotter
    from PyQt5.QtWidgets import QApplication
except ImportError:
    print("For real-time plotting, install: pip install pyvistaqt PyQt5")
    
def format_value(val):
    """Format values based on magnitude for annotations."""
    abs_val = abs(val)
    if abs_val == 0:
        return '0.0'
    elif abs_val < 0.01 or abs_val >= 1000:
        return f'{val:.3e}'
    else:
        return f'{val:.3f}'


class HexFEAPlotter:
    """Consolidated plotter for hex mesh FEA visualization."""
    
    def __init__(self, mesh, camera_position=None):
        """
        Initialize plotter with mesh.
        
        Args:
            mesh: HexMesher mesh object
            camera_position: Optional camera position [location, focal_point, up_vector]
        """
        self.mesh = mesh
        
        # Set default camera position if not provided
        if camera_position is None:
            view_distance = 2.5 * mesh.bbox.diag_length
            offset = 0.2 * view_distance
            self.camera_position = [
                (view_distance*0.5, -view_distance*0.3, view_distance),
                (offset, offset, 0),
                (0, 0.8, 0.4)
            ]
        else:
            self.camera_position = camera_position
        
        # Initialize standard plotter (can be overridden)
        self.pv_plotter = None
    
    # ========================================================================
    # MESH VISUALIZATION
    # ========================================================================
    
    def plot_mesh(self, voxels, num_components=1, plot_stl=False, 
                     stl_mesh=None, plotter=None):
        """
    Plot voxelized mesh with optional STL overlay.
    
    Args:
        voxels: PyVista mesh of voxels
        num_components: Number of mesh components
        plot_stl: Whether to show STL overlay
        stl_mesh: STL mesh for overlay
        plotter: Optional external plotter
    """
        if plotter is None:
            plotter = pv.Plotter(window_size=[400, 300])
        
        # Add voxelized mesh with component colors
        if num_components == 1:
            # Simple case - just plot the voxels
            plotter.add_mesh(voxels, show_edges=True)
        else:
            # Multi-component case - plot each component with different colors
            for i in range(num_components):
                color = [i/num_components, 1.0, 1 - i/num_components]
                component_cells = voxels.threshold(i + 1, scalars="component_id")
                if component_cells.n_cells > 0:
                    plotter.add_mesh(component_cells, color=color, 
                                label=f'Component {i}', show_edges=True)
            plotter.add_legend()
        
        plotter.add_axes()
        plotter.show_grid()
        
        if plot_stl and stl_mesh is not None and stl_mesh.n_faces > 0:
            # Add the original STL mesh for reference
            plotter.add_mesh(stl_mesh, color='red', show_edges=True, 
                            opacity=0.1, label='Original STL')
        
        plotter.show()
        
    def plot_mesh_structural(self, bc, title=None, plot_bc=True, 
                           rel_arrow_scale=0.1, offsetArrow=False,
                           save_path=None, plotter=None):
        """Plot structural mesh with boundary conditions."""
        external_plotter = plotter is not None
        
        if plotter is None:
            if save_path is not None:
                # Off-screen rendering for saving
                plotter = pv.Plotter(off_screen=True)
                plotter.camera_position = self.camera_position
                plotter.enable_anti_aliasing()
            else:
                # Interactive rendering
                plotter = pv.Plotter(window_size=[400,300])
        
        # Extract only surface faces for efficiency (important for large models)
        vertices = self.mesh.node_xyz
        faces, face_densities = self._extract_surface_faces(vertices)
        
        if len(faces) == 0:
            print("No faces to plot after filtering")
            return None
        
        # Create mesh with surface faces only
        n_faces = len(faces)
        cells = np.hstack((np.full((n_faces, 1), 4), faces))
        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices)
        pv_mesh.cell_data['density'] = face_densities
        
        plotter.add_mesh(pv_mesh, show_edges=True, color='lightgreen', 
                        edge_color='black', line_width=1)
        
        if plot_bc:
            self._add_structural_boundary_conditions(plotter, bc, rel_arrow_scale, offsetArrow)
        
        if title:
            plotter.add_title(title, font_size=8)
        
        plotter.camera_position = self.camera_position
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            if not external_plotter:
                plotter.show()
        
        self.camera_position = plotter.camera_position
    
    def plot_mesh_thermal(self, bc, title=None, plot_bc=True,
                         auto_close=True, save_path=None, plotter=None):
        """Plot thermal mesh with boundary conditions."""
        external_plotter = plotter is not None
        if plotter is None:
            if save_path is not None:
                # Off-screen rendering for saving
                plotter = pv.Plotter(off_screen=True)
                plotter.camera_position = self.camera_position
                plotter.enable_anti_aliasing()
            else:
                # Interactive rendering
                plotter = pv.Plotter(window_size=[400,300])
        
        # Extract only surface faces for efficiency (important for large models)
        vertices = self.mesh.node_xyz
        faces, face_densities = self._extract_surface_faces(vertices)
        
        if len(faces) == 0:
            print("No faces to plot after filtering")
            return None
        
        # Create mesh with surface faces only
        n_faces = len(faces)
        cells = np.hstack((np.full((n_faces, 1), 4), faces))
        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices)
        pv_mesh.cell_data['density'] = face_densities
        
        plotter.add_mesh(pv_mesh, show_edges=True, color='lightgreen',
                        edge_color='black', line_width=1)
        
        if plot_bc:
            self._add_thermal_boundary_conditions(plotter, bc)
        
        if title:
            plotter.add_title(title, font_size=8)
        
        plotter.camera_position = self.camera_position
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            if external_plotter:
                plotter.show(interactive_update=not auto_close, auto_close=auto_close)
            else:
                plotter.show()
        
        self.camera_position = plotter.camera_position
    
    # ========================================================================
    # STRUCTURAL RESULTS
    # ========================================================================
    
    def plot_structural_deformation(self, displacement, show_geometry=False,
                                    auto_close=True, save_path=None, 
                                    plotter=None, stl_mesh=None):
        """Plot structural deformation field."""
        # Reshape displacement to (num_nodes, 3)
        sol = displacement.reshape((-1, 3))
        
        # Compute deformation magnitude
        deformation = np.sqrt(sol[:, 0]**2 + sol[:, 1]**2 + sol[:, 2]**2)
        max_deformation = np.max(deformation)
        
        # Scale factor for visualization
        scale = float(0.1 * self.mesh.bbox.diag_length / max_deformation) if max_deformation > 0 else 1.0
        
        # Apply deformation to vertices
        vertices = self.mesh.node_xyz + scale * sol
        
        # Create mesh with only surface faces
        faces, face_densities = self._extract_surface_faces(vertices)
        
        if len(faces) == 0:
            print("No faces to plot")
            return
        
        # Create PyVista mesh
        n_faces = len(faces)
        cells = np.hstack((np.full((n_faces, 1), 4), faces))
        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices)
        pv_mesh.point_data['deformation'] = deformation
        
        # Plot
        external_plotter = plotter is not None
        if plotter is None:
            if save_path is not None:
                # Off-screen rendering for saving
                plotter = pv.Plotter(off_screen=True)
                plotter.camera_position = self.camera_position
                plotter.enable_anti_aliasing()
            else:
                # Interactive rendering
                plotter = pv.Plotter(window_size=[400,300])
        
        plotter.add_mesh(pv_mesh, scalars='deformation', show_edges=True,
                        cmap='jet', edge_color='black', line_width=1,
                        scalar_bar_args={'title': 'Deformation', 'vertical': True})
        
        if show_geometry and stl_mesh is not None:
            plotter.add_mesh(stl_mesh, opacity=0.5, color='white', show_edges=True)
        
        plotter.add_title(f'Deformation (scaled {scale:.2f}x)', font_size=8)
        plotter.camera_position = self.camera_position
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            if external_plotter:
                plotter.show(interactive_update=not auto_close, auto_close=auto_close)
            else:
                plotter.show()
        
        self.camera_position = plotter.camera_position
    
    def plot_von_mises_stress(self, von_mises_stress, save_path=None, 
                             fontsize=8, plotter=None):
        """Plot von Mises stress field."""
        self.plot_elem_field(von_mises_stress, title='von Mises Stress',
                            save_path=save_path, fontsize=fontsize, plotter=plotter)
    
    def plot_stress_component(self, stress_component, component_idx=0,
                             save_path=None, fontsize=10):
        """Plot specific stress component."""
        self.plot_elem_field(stress_component, title=f'Stress Component {component_idx}',
                            save_path=save_path, fontsize=fontsize)
    
    def plot_strain_component(self, strain_component, component_idx=0,
                             save_path=None, fontsize=8):
        """Plot specific strain component."""
        self.plot_elem_field(strain_component, title=f'Strain Component {component_idx}',
                            save_path=save_path, fontsize=fontsize)
    
    # ========================================================================
    # THERMAL RESULTS
    # ========================================================================
    
    def plot_temperature(self, temperature, auto_close=True, plotter=None,
                        save_path=None, annotate_max_min=False):
        """Plot temperature field."""
        # Create deformed mesh with temperature
        sol = temperature.reshape((-1, 1))
        vertices = self.mesh.node_xyz.copy()
        
        # Extract surface faces
        faces, face_densities = self._extract_surface_faces(vertices)
        
        if len(faces) == 0:
            print("No faces to plot")
            return
        
        # Create PyVista mesh
        n_faces = len(faces)
        cells = np.hstack((np.full((n_faces, 1), 4), faces))
        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices)
        
        # Set point data for temperature
        pv_mesh.point_data['temperature'] = sol
        
        # Plot
        external_plotter = plotter is not None
        if plotter is None:
            if save_path is not None:
                # Off-screen rendering for saving
                plotter = pv.Plotter(off_screen=True)
                plotter.camera_position = self.camera_position
                plotter.enable_anti_aliasing()
            else:
                # Interactive rendering
                plotter = pv.Plotter( window_size=[400,300])
        
        plotter.add_mesh(pv_mesh, scalars='temperature', show_edges=True,
                        cmap='jet', edge_color='black', line_width=1,
                        scalar_bar_args={'title': 'Temperature', 'vertical': True})
        
        if annotate_max_min:
            # Add annotations for max/min
            temp_vals = sol.flatten()
            max_idx = np.argmax(temp_vals)
            min_idx = np.argmin(temp_vals)
            plotter.add_point_labels([vertices[max_idx]], [f'Max: {format_value(temp_vals[max_idx])}'],
                                    point_size=10, font_size=14, text_color='red')
            plotter.add_point_labels([vertices[min_idx]], [f'Min: {format_value(temp_vals[min_idx])}'],
                                    point_size=10, font_size=14, text_color='blue')
        
        plotter.add_title('Temperature Field', font_size=8)
        plotter.camera_position = self.camera_position
        plotter.add_axes()
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            if external_plotter:
                plotter.show(interactive_update=not auto_close, auto_close=auto_close)
            else:
                plotter.show()
        
        self.camera_position = plotter.camera_position
    
    # ========================================================================
    # MODAL RESULTS
    # ========================================================================
    
    def plot_eigenmode(self, eigenvector, eigenvalue, mode_number=0, plotter=None):
        """Plot eigenmode shape."""
        # Reshape eigenvector
        sol = eigenvector.reshape((-1, 3))
        
        # Scale deformation for visualization
        deltaMax = np.max(np.abs(sol))
        scale = float(0.1 * self.mesh.bbox.diag_length / deltaMax) if deltaMax > 0 else 1.0
        vertices = self.mesh.node_xyz + scale * sol
        
        # Extract surface faces
        faces, face_densities = self._extract_surface_faces(vertices)
        
        if len(faces) == 0:
            print("No faces to plot")
            return
        
        # Create PyVista mesh
        n_faces = len(faces)
        cells = np.hstack((np.full((n_faces, 1), 4), faces))
        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices)
        pv_mesh.point_data['values'] = sol
        
        # Plot
        if plotter is None:
            plotter = pv.Plotter( window_size=[400,300])
        
        plotter.add_mesh(pv_mesh, scalars='values', show_edges=True,
                        cmap='jet', edge_color='black', line_width=1,
                        scalar_bar_args={'title': '', 'vertical': True})
        
        plotter.add_title(f'Eigenmode {mode_number}; freq: {eigenvalue:.3g} Hz', font_size=8)
        plotter.camera_position = self.camera_position
        plotter.add_axes()
        plotter.camera.zoom(0.8)
        plotter.enable_anti_aliasing()
        plotter.show()
    
    # ========================================================================
    # GENERIC ELEMENT FIELD PLOTTING
    # ========================================================================
    
    def plot_elem_field(self, elem_field, title='Field', colormap='jet',
                       mask_low_pseudodensity=True, auto_close=True,
                       save_path=None, fontsize=10, plotter=None,
                       annotate_max_min=True, show_geometry=False,
                       cross_section=None, filtered_colors=None):
        """
        Generic element field plotting (stress, strain, density, etc.).
        
        Args:
            elem_field: Array of element field values
            title: Plot title
            colormap: Color map to use
            mask_low_pseudodensity: Filter elements below 0.01 density
            auto_close: Whether to close plot automatically
            save_path: Path to save screenshot
            fontsize: Font size for title
            plotter: External plotter to use
            annotate_max_min: Add min/max annotations
            show_geometry: Show original STL geometry
            cross_section: Tuple (axis, position) for cross-section
            filtered_colors: Optional RGB colors for each element
        """
        external_plotter = plotter is not None
        if plotter is None:
            if save_path is not None:
                # Off-screen rendering for saving
                plotter = pv.Plotter(off_screen=True)
                plotter.camera_position = self.camera_position
                plotter.enable_anti_aliasing()
            else:
                # Interactive rendering
                plotter = pv.Plotter()
        
        # Filter elements based on density
        if mask_low_pseudodensity:
            mask = self.mesh.elemPseudoDensity > 0.01
            filtered_elems = self.mesh.elemArray[mask]
            filtered_field = elem_field[mask]
        else:
            filtered_elems = self.mesh.elemArray
            filtered_field = elem_field
        
        if len(filtered_elems) == 0:
            print("No elements to plot after filtering")
            return
        
        # Create PyVista mesh
        cells = np.hstack((np.full((len(filtered_elems), 1), 8), filtered_elems))
        pv_mesh = pv.UnstructuredGrid({pv.CellType.HEXAHEDRON: cells[:, 1:]}, 
                                      self.mesh.node_xyz)
        pv_mesh.cell_data['field'] = filtered_field
        
        # Apply cross-section if requested
        if cross_section:
            axis, position = cross_section
            if axis == 'x':
                normal, origin = (1, 0, 0), (position, 0, 0)
            elif axis == 'y':
                normal, origin = (0, 1, 0), (0, position, 0)
            elif axis == 'z':
                normal, origin = (0, 0, 1), (0, 0, position)
            else:
                normal, origin = (1, 0, 0), (0, 0, 0)
            pv_mesh = pv_mesh.clip(normal=normal, origin=origin)
        
        # Add mesh to plotter
        if filtered_colors is not None:
            plotter.add_mesh(pv_mesh, scalars=filtered_colors, rgb=True,
                           show_edges=True, edge_color='black', line_width=1)
        else:
            plotter.add_mesh(pv_mesh, scalars='field', cmap=colormap,
                           show_edges=True, edge_color='black', line_width=1,
                           scalar_bar_args={'title': '', 'vertical': True})
        
        # Annotate max/min
        if annotate_max_min and len(filtered_field) > 0:
            max_idx = np.argmax(filtered_field)
            min_idx = np.argmin(filtered_field)
            cell_centers = pv_mesh.cell_centers().points
            plotter.add_point_labels([cell_centers[max_idx]], 
                                    [f'Max: {format_value(filtered_field[max_idx])}'],
                                    point_size=10, font_size=fontsize*2, text_color='red')
            plotter.add_point_labels([cell_centers[min_idx]],
                                    [f'Min: {format_value(filtered_field[min_idx])}'],
                                    point_size=10, font_size=fontsize*2, text_color='blue')
        
        if show_geometry and hasattr(self.mesh, 'stlGeom'):
            # Add STL geometry
            pass  # Implement if needed
        
        plotter.add_title(title, font_size=0.9*fontsize)
        plotter.camera_position = self.camera_position
        
        if save_path:
            plotter.screenshot(save_path)
            plotter.close()
        else:
            if external_plotter:
                plotter.show(interactive_update=not auto_close, auto_close=auto_close)
            else:
                plotter.show()
        
        self.camera_position = plotter.camera_position
    
    def plot_pseudo_density(self, save_path=None, auto_close=True,
                           title='Pseudo density', fontsize=10, plotter=None):
        """Plot pseudo density field."""
        if plotter is not None:
            plotter.clear()
        self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r',
                            auto_close=auto_close, mask_low_pseudodensity=False,
                            title=title, save_path=save_path, fontsize=fontsize,
                            plotter=plotter)
    
    # ========================================================================
    # REAL-TIME OPTIMIZATION VISUALIZATION
    # ========================================================================
    
    def plot_pseudo_density_realtime(self, title='Pseudo density', iteration=0):
        """
        Real-time non-blocking visualization for optimization.
        Uses BackgroundPlotter with proper Qt event loop handling.
        """
            
        if not hasattr(self, '_rt_plotter'):
            # Get or create QApplication
            app = QApplication.instance()
            if app is None:
                app = QApplication([])
            self._qt_app = app
            
            # Create BackgroundPlotter
            self._rt_plotter = BackgroundPlotter(window_size=(800, 600))
            self._rt_plotter.camera_position = self.camera_position
            
            # Create mesh
            cells = self.mesh.elemArray.shape[0]
            cell_type = np.full(cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
            cells_pv = np.column_stack((np.full(cells, 8), self.mesh.elemArray)).flatten()
            pv_mesh = pv.UnstructuredGrid(cells_pv, cell_type, self.mesh.node_xyz)
            
            density = self.mesh.elemPseudoDensity.copy()
            mask = density > 0.01
            pv_mesh = pv_mesh.extract_cells(mask)
            pv_mesh.cell_data['density'] = density[mask]
            
            self._rt_plotter.add_mesh(pv_mesh, scalars='density', cmap='gray_r',
                                        show_edges=True, edge_color='black', line_width=1)
            self._rt_plotter.add_text(title, position='upper_edge', font_size=10)
        else:
            # Update mesh
            cells = self.mesh.elemArray.shape[0]
            cell_type = np.full(cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
            cells_pv = np.column_stack((np.full(cells, 8), self.mesh.elemArray)).flatten()
            pv_mesh = pv.UnstructuredGrid(cells_pv, cell_type, self.mesh.node_xyz)
            
            density = self.mesh.elemPseudoDensity.copy()
            mask = density > 0.01
            pv_mesh = pv_mesh.extract_cells(mask)
            pv_mesh.cell_data['density'] = density[mask]
            
            self._rt_plotter.clear()
            self._rt_plotter.add_mesh(pv_mesh, scalars='density', cmap='gray_r',
                                        show_edges=True, edge_color='black', line_width=1)
            self._rt_plotter.add_text(title, position='upper_edge', font_size=10)
        
            # CRITICAL: Process Qt events!
        self._qt_app.processEvents()
            
 
    def save_pseudo_density_snapshot(self, filename, title='Pseudo density'):
        """Save snapshot without GUI (for optimization loops)."""
        cells = self.mesh.elemArray.shape[0]
        cell_type = np.full(cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)
        cells_pv = np.column_stack((np.full(cells, 8), self.mesh.elemArray)).flatten()
        pv_mesh = pv.UnstructuredGrid(cells_pv, cell_type, self.mesh.node_xyz)
        
        density = self.mesh.elemPseudoDensity.copy()
        mask = density > 0.01
        pv_mesh = pv_mesh.extract_cells(mask)
        pv_mesh.cell_data['density'] = density[mask]
        
        plotter = pv.Plotter(off_screen=True, window_size=[400,300])
        plotter.camera_position = self.camera_position
        plotter.add_mesh(pv_mesh, scalars='density', cmap='gray_r',
                        show_edges=True, edge_color='black', line_width=1)
        plotter.add_text(title, position='upper_edge', font_size=14)
        plotter.screenshot(filename)
        plotter.close()
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_surface_faces(self, vertices):
        """Extract surface faces from hex mesh."""
        faceIndex = np.array([[0,4,7,3], [0,1,5,4], [0,3,2,1],
                             [1,2,6,5], [2,3,7,6], [4,5,6,7]], dtype=np.uint32)
        
        faces = []
        face_densities = []
        
        for e in range(self.mesh.num_elems):
            if self.mesh.elemPseudoDensity[e] < 0.5:
                continue
            elif (self.mesh.elemPseudoDensity[e] > 0.5 and 
                  np.all(self.mesh.elemNeighborsArray[e] > 0) and
                  np.all(self.mesh.elemPseudoDensity[[int(elem) for elem in 
                                                     self.mesh.elemNeighborsArray[e]]] > 0.5)):
                continue
            
            for j in range(6):
                faces.append(self.mesh.elemArray[e, faceIndex[j, :]])
                face_densities.append(self.mesh.elemPseudoDensity[e])
        
        return np.array(faces) if faces else np.array([]), np.array(face_densities)
    
    def _add_structural_boundary_conditions(self, plotter, bc, rel_arrow_scale, offsetArrow):
        """
        Add structural boundary condition visualization.
        
        Args:
            plotter: PyVista plotter
            bc: Boundary condition object with force array and fixed DOFs
            rel_arrow_scale: Relative scale for force arrows
            offsetArrow: If True, offset arrow start point for visibility
        """

        vertices = self.mesh.node_xyz
        
        # Add black spheres for label 1 (fixed nodes)
        label1_nodes = np.where(self.mesh.node_indices[:, 3] == 1)[0]
        
        if len(label1_nodes) > 0 and bc is not None:
            points1 = vertices[label1_nodes]
            pts = pv.PointSet(points1)
            plotter.add_mesh(pts, color='black', point_size=10, 
                            render_points_as_spheres=True)
        
        # Add red force arrows for label 2 (loaded nodes)
        label2_nodes = np.where(self.mesh.node_indices[:, 3] == 2)[0]
        
        if len(label2_nodes) > 0 and bc is not None:
            # Calculate average force norm for scaling
            force_norms = []
            for node in label2_nodes:
                fx = bc.force[3*node]
                fy = bc.force[3*node + 1]
                fz = bc.force[3*node + 2]
                force_vec = np.array([fx, fy, fz])
                force_norm = np.linalg.norm(force_vec)
                if force_norm > 0:
                    force_norms.append(force_norm)
            
            if len(force_norms) > 0:
                force_norm_avg = np.mean(force_norms)
                
                # Add arrows for each loaded node
                for node in label2_nodes:
                    # Get force components for this node
                    fx = bc.force[3*node]
                    fy = bc.force[3*node + 1]
                    fz = bc.force[3*node + 2]
                    force_vec = np.array([fx, fy, fz])
                    
                    force_norm = np.linalg.norm(force_vec)
                    
                    # Only add arrow if force is non-zero
                    if force_norm > 0:
                        # Scale arrow based on relative force magnitude
                        arrow_scale = rel_arrow_scale * self.mesh.bbox.diag_length * \
                                    force_norm / force_norm_avg
                        
                        # Normalize force vector for direction
                        force_vec_dir = force_vec / force_norm
                        
                        # Determine start point
                        start_point = vertices[node].copy()
                        if offsetArrow:
                            # Offset start point so arrow is visible outside mesh
                            start_point = start_point - force_vec_dir * arrow_scale
                        
                        # Create and add arrow
                        arrow = pv.Arrow(start=start_point,
                                        direction=force_vec_dir,
                                        scale=arrow_scale)
                        plotter.add_mesh(arrow, color='red')


    def _add_thermal_boundary_conditions(self, plotter, bc):

        if bc is None:
            return
        
        vertices = self.mesh.node_xyz
        point_size = 10
        
        # For thermal problems, BC info is in bc.fixed_dofs, not node labels
        if not hasattr(bc, 'fixed_dofs') or bc.fixed_dofs is None:
            return
        
        # Get nodes with prescribed temperatures
        # fixed_dofs are DOF indices, convert to node indices
        fixed_nodes = bc.fixed_dofs  # Already node indices for thermal (1 DOF per node)
        
        if len(fixed_nodes) > 0:
            # Separate by temperature value
            # Nodes with temp = 0 (cold)
            cold_mask = bc.dirichlet_values == 0
            cold_nodes = fixed_nodes[cold_mask]
            
            # Nodes with temp > 0 (hot)
            hot_mask = bc.dirichlet_values > 0
            hot_nodes = fixed_nodes[hot_mask]
            
            # Add cold nodes as blue spheres
            if len(cold_nodes) > 0:
                points_cold = vertices[cold_nodes]
                dots_cold = pv.PolyData(points_cold)
                plotter.add_points(dots_cold,
                                color='blue',
                                point_size=point_size,
                                render_points_as_spheres=True)
            
            # Add hot nodes as red spheres
            if len(hot_nodes) > 0:
                points_hot = vertices[hot_nodes]
                dots_hot = pv.PolyData(points_hot)
                plotter.add_points(dots_hot,
                                color='red',
                                point_size=point_size,
                                render_points_as_spheres=True)