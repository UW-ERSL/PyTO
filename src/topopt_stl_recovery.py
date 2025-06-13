import pyvista as pv
import numpy as np
import os
import traceback
import enum
import trimesh
import pymeshfix


## get rid of this  class
class BoundingBoxAlignment:
    """Helper class for bounding box analysis and alignment operations"""
    
    @staticmethod
    def get_bounds_info(mesh, name="Mesh"):
        bounds = mesh.bounds
        center = mesh.center
        x_size = bounds[1] - bounds[0]
        y_size = bounds[3] - bounds[2]
        z_size = bounds[5] - bounds[4]
        info = {
            'name': name,
            'bounds': bounds,
            'center': center,
            'size': [x_size, y_size, z_size],
            'volume': x_size * y_size * z_size,
            'diagonal': np.sqrt(x_size**2 + y_size**2 + z_size**2)
        }
        return info
    
    @staticmethod
    def print_bounds_comparison(stl_info, voids_info):
        center_offset = np.array(voids_info['center']) - np.array(stl_info['center'])
        size_ratio = np.array(voids_info['size']) / np.array(stl_info['size'])
        stl_bounds = stl_info['bounds']
        void_bounds = voids_info['bounds']
        x_contained = stl_bounds[0] <= void_bounds[0] and void_bounds[1] <= stl_bounds[1]
        y_contained = stl_bounds[2] <= void_bounds[2] and void_bounds[3] <= stl_bounds[3]
        z_contained = stl_bounds[4] <= void_bounds[4] and void_bounds[5] <= stl_bounds[5]
        fully_contained = x_contained and y_contained and z_contained
        return {
            'center_offset': center_offset,
            'size_ratio': size_ratio,
            'fully_contained': fully_contained,
            'center_distance': np.linalg.norm(center_offset)
        }
    
    @staticmethod
    def align_voids_to_stl(void_mesh, stl_info, voids_info, alignment_method='smart'):
        aligned_mesh = void_mesh.copy()
        if alignment_method == 'center_only':
            center_offset = np.array(stl_info['center']) - np.array(voids_info['center'])
            aligned_mesh = aligned_mesh.translate(center_offset)
            alignment_info = {
                'method': 'center_only',
                'translation': center_offset,
                'scale_factor': [1.0, 1.0, 1.0],
                'final_bounds': aligned_mesh.bounds
            }
        elif alignment_method == 'scale_only':
            scale_factor = np.array(stl_info['size']) / np.array(voids_info['size'])
            void_center = np.array(voids_info['center'])
            aligned_mesh = aligned_mesh.translate(-void_center)
            aligned_mesh = aligned_mesh.scale(scale_factor)
            aligned_mesh = aligned_mesh.translate(void_center)
            alignment_info = {
                'method': 'scale_only',
                'translation': [0.0, 0.0, 0.0],
                'scale_factor': scale_factor,
                'final_bounds': aligned_mesh.bounds
            }
        elif alignment_method == 'fit_to_stl':
            scale_factor = np.array(stl_info['size']) / np.array(voids_info['size'])
            center_offset = np.array(stl_info['center']) - np.array(voids_info['center'])
            void_center = np.array(voids_info['center'])
            aligned_mesh = aligned_mesh.translate(-void_center)
            aligned_mesh = aligned_mesh.scale(scale_factor)
            aligned_mesh = aligned_mesh.translate(stl_info['center'])
            alignment_info = {
                'method': 'fit_to_stl',
                'translation': center_offset,
                'scale_factor': scale_factor,
                'final_bounds': aligned_mesh.bounds
            }
        elif alignment_method == 'smart':
            center_distance = np.linalg.norm(np.array(voids_info['center']) - np.array(stl_info['center']))
            size_ratio = np.array(voids_info['size']) / np.array(stl_info['size'])
            avg_size_ratio = np.mean(size_ratio)
            if center_distance > max(stl_info['size']) * 0.1:
                if abs(avg_size_ratio - 1.0) > 0.2:
                    return BoundingBoxAlignment.align_voids_to_stl(void_mesh, stl_info, voids_info, 'fit_to_stl')
                else:
                    return BoundingBoxAlignment.align_voids_to_stl(void_mesh, stl_info, voids_info, 'center_only')
            elif abs(avg_size_ratio - 1.0) > 0.2:
                return BoundingBoxAlignment.align_voids_to_stl(void_mesh, stl_info, voids_info, 'scale_only')
            else:
                center_offset = np.array(stl_info['center']) - np.array(voids_info['center'])
                aligned_mesh = aligned_mesh.translate(center_offset * 0.5)
                alignment_info = {
                    'method': 'smart_minor',
                    'translation': center_offset * 0.5,
                    'scale_factor': [1.0, 1.0, 1.0],
                    'final_bounds': aligned_mesh.bounds
                }
        else:
            return BoundingBoxAlignment.align_voids_to_stl(void_mesh, stl_info, voids_info, 'fit_to_stl')
        return aligned_mesh, alignment_info
    
    @staticmethod
    def scale_stl_to_voids(stl_mesh, stl_info, voids_info, padding_factor=1.1):
        required_scale = np.array(voids_info['size']) / np.array(stl_info['size'])
        required_scale *= padding_factor
        uniform_scale = np.max(required_scale)
        stl_center = np.array(stl_info['center'])
        scaled_stl = stl_mesh.copy()
        scaled_stl = scaled_stl.translate(-stl_center)
        scaled_stl = scaled_stl.scale([uniform_scale, uniform_scale, uniform_scale])
        scaled_stl = scaled_stl.translate(stl_center)
        scaling_info = {
            'method': 'scale_stl_to_voids',
            'scale_factor': uniform_scale,
            'padding_factor': padding_factor,
            'original_stl_bounds': stl_mesh.bounds,
            'scaled_stl_bounds': scaled_stl.bounds
        }
        return scaled_stl, scaling_info


class ExamplesCAD(enum.Enum):
    Mitchell_1 = enum.auto()
    Mitchell_2 = enum.auto()
    TwoBar = enum.auto()
    MBBB = enum.auto()
    DistributedLoad = enum.auto()
    LBracketMidLoad = enum.auto()
    VerticalBar = enum.auto()
    CircularPlateHole = enum.auto()
    KnuckleAssembly = enum.auto()
    ShortCantileverMidLoad = enum.auto()
    CantileverMidLoad = enum.auto()
    ThreeHoleBracket = enum.auto()


def get_example_cad(example: ExamplesCAD):
    modelName = example.name.split('_')[0]
    fp_original_stl =  f"Models/{modelName}/{modelName}.STL"
    fp_vtu_mesh  = f"Results/VTU/{example.name}.vtu"
    fp_outputstlpath = f"Results/VTU/{example.name}_result.stl"
    return fp_original_stl, fp_vtu_mesh, fp_outputstlpath


class TopOptSTLRecoveryWithAlignment:

    def __init__(self, visualize=False, decimation_target_ratio=0.3, max_triangles_per_patch=10000, 
                 alignment_method='smart', scale_approach='align_voids', **kwargs):
        self.visualize = visualize
        self.decimation_target_ratio = decimation_target_ratio
        self.max_triangles_per_patch = max_triangles_per_patch
        self.alignment_method = alignment_method
        self.scale_approach = scale_approach
        self.bbox_helper = BoundingBoxAlignment()

    def find_and_prepare_density_field(self, vtu_mesh):
        if vtu_mesh.cell_data:
            for key, data in vtu_mesh.cell_data.items():
                pass
        if vtu_mesh.point_data:
            for key, data in vtu_mesh.point_data.items():
                pass
        possible_names = [
            'density', 'Density', 'DENSITY',
            'rho', 'Rho', 'RHO',
            'x', 'X',
            'pseudo_density', 'PseudoDensity',
            'element_density', 'ElementDensity',
            'material_density', 'MaterialDensity',
            'design_variable', 'DesignVariable',
            'topopt_density', 'TopOptDensity',
            'elemPseudoDensity'
        ]
        density_data = None
        density_field_name = None
        data_location = None
        if vtu_mesh.point_data:
            for field_name in possible_names:
                if field_name in vtu_mesh.point_data:
                    data = vtu_mesh.point_data[field_name]
                    if len(data.shape) == 1:
                        density_data = data
                        density_field_name = field_name
                        data_location = 'point'
                        break
            if density_data is None:
                for field_name, data in vtu_mesh.point_data.items():
                    if len(data.shape) == 1:
                        min_val, max_val = np.min(data), np.max(data)
                        if 0 <= min_val <= 1.1 and 0 <= max_val <= 1.1:
                            density_data = data
                            density_field_name = field_name
                            data_location = 'point'
                            break
        if density_data is None and vtu_mesh.cell_data:
            for field_name in possible_names:
                if field_name in vtu_mesh.cell_data:
                    data = vtu_mesh.cell_data[field_name]
                    if len(data.shape) == 1:
                        density_data = data
                        density_field_name = field_name
                        data_location = 'cell'
                        break
            if density_data is None:
                for field_name, data in vtu_mesh.cell_data.items():
                    if len(data.shape) == 1:
                        min_val, max_val = np.min(data), np.max(data)
                        if 0 <= min_val <= 1.1 and 0 <= max_val <= 1.1:
                            density_data = data
                            density_field_name = field_name
                            data_location = 'cell'
                            break
        if density_data is None:
            print("❌ No suitable density field found!")
            return None, None
        mesh_copy = vtu_mesh.copy()
        min_val, max_val = np.min(density_data), np.max(density_data)
        data_range = max_val - min_val
        needs_normalization = (
            min_val < -0.1 or
            max_val > 1.1 or
            data_range < 0.1 or
            min_val > 0.9
        )
        if needs_normalization:
            normalized_density = (density_data - min_val) / (max_val - min_val)
        else:
            normalized_density = density_data
        if data_location == 'point':
            mesh_copy.point_data['normalized_density'] = normalized_density
            mesh_copy = mesh_copy.point_data_to_cell_data()
        else:
            mesh_copy.cell_data['normalized_density'] = normalized_density
        final_density = mesh_copy.cell_data['normalized_density']
        return mesh_copy, 'normalized_density'

    def ensure_polydata(self, mesh, name="mesh"):
        if isinstance(mesh, pv.PolyData):
            return mesh
        try:
            if hasattr(mesh, 'extract_surface'):
                poly_mesh = mesh.extract_surface()
                if isinstance(poly_mesh, pv.PolyData):
                    return poly_mesh
            if hasattr(mesh, 'faces') and hasattr(mesh, 'points'):
                poly_mesh = pv.PolyData(mesh.points, faces=mesh.faces)
                return poly_mesh
            print(f"Warning: Could not convert {name} to PolyData, using original")
            return mesh
        except Exception as e:
            print(f"Error converting {name} to PolyData: {e}")
            return mesh

    def decimate_mesh(self, pv_mesh, target_ratio=None, max_triangles=None):
        if target_ratio is None:
            target_ratio = self.decimation_target_ratio
        if max_triangles is None:
            max_triangles = self.max_triangles_per_patch
        pv_mesh = self.ensure_polydata(pv_mesh, "decimation input")
        if not hasattr(pv_mesh, 'is_all_triangles') or not pv_mesh.is_all_triangles:
            pv_mesh = pv_mesh.triangulate()
        original_faces = pv_mesh.n_cells
        target_from_ratio = int(original_faces * target_ratio)
        target_faces = min(target_from_ratio, max_triangles)
        if original_faces <= target_faces:
            return pv_mesh
        try:
            reduction_ratio = 1.0 - (target_faces / original_faces)
            decimated = pv_mesh.decimate(reduction_ratio)
            return decimated
        except Exception as e:
            print(f"Decimation failed: {e}")
            return pv_mesh

    def to_trimesh(self, pv_mesh):
        try:
            if pv_mesh.n_cells == 0:
                print("Warning: Mesh has no cells")
                return None
            pv_mesh = self.ensure_polydata(pv_mesh, "trimesh conversion")
            if not hasattr(pv_mesh, 'is_all_triangles') or not pv_mesh.is_all_triangles:
                pv_mesh = pv_mesh.triangulate()
            faces_array = pv_mesh.faces
            if faces_array is None or len(faces_array) == 0:
                print("Warning: Mesh has no faces array")
                return None
            try:
                faces = faces_array.reshape(-1, 4)[:, 1:4]
            except ValueError as e:
                print(f"Error reshaping faces array: {e}")
                print(f"Faces array shape: {faces_array.shape}")
                print(f"First 20 elements: {faces_array[:20] if len(faces_array) >= 20 else faces_array}")
                return None
            tm = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)
            if len(tm.faces) == 0:
                print("Warning: Created trimesh has no faces")
                return None
            return tm
        except Exception as e:
            print(f"Error converting to trimesh: {e}")
            traceback.print_exc()
            return None

    def extract_isocontour_surface(self, vtu_mesh, isovalue=0.5, resolution=1, 
                                  binarization=False, density_field='normalized_density'):
        vertices = vtu_mesh.points
        if density_field in vtu_mesh.cell_data:
            elemPseudoDensity = vtu_mesh.cell_data[density_field]
        else:
            print(f"Warning: {density_field} not found in cell_data")
            elemPseudoDensity = np.ones(vtu_mesh.n_cells)
        if binarization:
            elemPseudoDensity = np.where(elemPseudoDensity > 0.5, 1, 0)
        pv_mesh = vtu_mesh.copy()
        pv_mesh.cell_data[density_field] = elemPseudoDensity
        mesh_with_point_data = pv_mesh.cell_data_to_point_data()
        bounds = pv_mesh.bounds
        
        ## Add human comments
        ## Explain each parameter

        print(f"VTU mesh bounds: {bounds}")
        padding = max([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]) * 0.1
        padding = min([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4], padding])
        nVoxels = resolution *pv_mesh.n_cells
        #nx = alpha*bounds[1]-bounds[0], ny = alpha*bounds[3]-bounds[2], nz = alpha*bounds[5]-bounds[4]
        resolution = 1
        alpha = (nVoxels / np.prod(np.array(bounds[1::2]) - np.array(bounds[0::2])))**(1/3)
        
        dimensions = int(alpha * (bounds[1] - bounds[0]) ), int(alpha * (bounds[3] - bounds[2]) ),  int(alpha * (bounds[5] - bounds[4]) )   
        print(f"Creating grid with dimensions: {dimensions} and padding: {padding}")

        spacing = (
            (bounds[1] - bounds[0] + 2*padding) / (dimensions[0] - 1),
            (bounds[3] - bounds[2] + 2*padding) / (dimensions[1] - 1),
            (bounds[5] - bounds[4] + 2*padding) / (dimensions[2] - 1)
        )
        origin = (bounds[0] - padding, bounds[2] - padding, bounds[4] - padding)
        grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)
        grid_with_data = grid.interpolate(mesh_with_point_data, radius=padding/3, null_value=-1)
        isosurf = grid_with_data.contour([isovalue], scalars=density_field)
        return isosurf

    def create_stl_from_topopt_result(self, design_domain_stl_filepath=None, topopt_result_vtu_filepath=None, 
                            isovalue=0.5, resolution=1, binarization=True):
        if not os.path.exists(design_domain_stl_filepath):
            print(f"Error: Input STL file not found at {design_domain_stl_filepath}")
            return None
        if not os.path.exists(topopt_result_vtu_filepath):
            print(f"Error: Input VTU file not found at {topopt_result_vtu_filepath}")
            return None
        try:
            voxels_original = pv.read(topopt_result_vtu_filepath)

            ## Can we simplify this?
            voxels, density_field_name = self.find_and_prepare_density_field(voxels_original)
            if voxels is None:
                print("❌ No suitable density field found in VTU mesh!")
            
            ## Why are we triangulating again?    
            original_stl = pv.read(design_domain_stl_filepath).triangulate().compute_normals()
            voxels_inverted = voxels.copy()
            voxels_inverted.cell_data['inverted_density'] = 1.0 - voxels.cell_data[density_field_name]
            if self.visualize:
                p = pv.Plotter()
                p.add_mesh(original_stl, color='lightblue', opacity=0.3, label='Original STL')
                p.add_mesh(voxels_inverted, scalars='inverted_density', cmap='plasma', opacity=0.5, label='Inverted Density')
                p.add_title("Inverted (Negative) Density Field")
                p.add_legend()
                p.camera_position = 'iso'
                p.show()


            # Make this step robust for all examples (2.5D, 3D, etc.)
            void_iso_surface = self.extract_isocontour_surface(
                voxels_inverted, 
                isovalue=isovalue, 
                resolution=resolution, 
                binarization=binarization,
                density_field='inverted_density'
            )

            ## Get rid of this  
            if void_iso_surface is None or void_iso_surface.n_cells == 0:
                return original_stl
            vtu_bounds_info = self.bbox_helper.get_bounds_info(voxels, "VTU Mesh")
            stl_info = self.bbox_helper.get_bounds_info(original_stl, "Original STL")
            voids_info = self.bbox_helper.get_bounds_info(void_iso_surface, "Void Regions")
            alignment_analysis = self.bbox_helper.print_bounds_comparison(stl_info, voids_info)
            stl_to_vtu_scale = np.array(vtu_bounds_info['size']) / np.array(stl_info['size'])
            stl_to_vtu_center_offset = np.array(vtu_bounds_info['center']) - np.array(stl_info['center'])
            final_stl = original_stl
            final_voids = void_iso_surface
            if self.scale_approach == 'scale_stl':
                vtu_bounds = vtu_bounds_info['bounds']
                stl_bounds = stl_info['bounds']
                scale_x = (vtu_bounds[1] - vtu_bounds[0]) / (stl_bounds[1] - stl_bounds[0])
                scale_y = (vtu_bounds[3] - vtu_bounds[2]) / (stl_bounds[3] - stl_bounds[2])
                scale_z = (vtu_bounds[5] - vtu_bounds[4]) / (stl_bounds[5] - stl_bounds[4])
                stl_center = np.array(stl_info['center'])
                final_stl = original_stl.copy()
                final_stl = final_stl.translate(-stl_center)
                final_stl = final_stl.scale([scale_x, scale_y, scale_z])
                vtu_center = np.array(vtu_bounds_info['center'])
                final_stl = final_stl.translate(vtu_center)
                final_size = np.array(stl_info['size']) * np.array([scale_x, scale_y, scale_z])
                size_diff = np.abs(final_size - np.array(vtu_bounds_info['size']))
                perfect_match = np.all(size_diff < 1e-6)
                stl_info = self.bbox_helper.get_bounds_info(final_stl, "Perfectly Scaled STL")
                final_stl_bounds = final_stl.bounds
                vtu_bounds_diff = np.abs(np.array(final_stl_bounds) - np.array(vtu_bounds))
                bounds_match = np.max(vtu_bounds_diff) < 1e-6
            else:
                if not alignment_analysis['fully_contained'] or alignment_analysis['center_distance'] > max(stl_info['size']) * 0.05:
                    final_voids, alignment_info = self.bbox_helper.align_voids_to_stl(
                        void_iso_surface, stl_info, voids_info, self.alignment_method
                    )
                    voids_info = self.bbox_helper.get_bounds_info(final_voids, "Aligned Voids")
            final_stl_info = self.bbox_helper.get_bounds_info(final_stl, "Final STL")
            final_voids_info = self.bbox_helper.get_bounds_info(final_voids, "Final Voids")
            if self.scale_approach == 'scale_stl':
                vtu_coverage = (
                    final_stl_info['bounds'][0] <= vtu_bounds_info['bounds'][0] and
                    final_stl_info['bounds'][1] >= vtu_bounds_info['bounds'][1] and
                    final_stl_info['bounds'][2] <= vtu_bounds_info['bounds'][2] and
                    final_stl_info['bounds'][3] >= vtu_bounds_info['bounds'][3] and
                    final_stl_info['bounds'][4] <= vtu_bounds_info['bounds'][4] and
                    final_stl_info['bounds'][5] >= vtu_bounds_info['bounds'][5]
                )
            final_alignment = self.bbox_helper.print_bounds_comparison(final_stl_info, final_voids_info)
            if self.visualize:
                p = pv.Plotter(shape=(1, 2))
                p.subplot(0, 0)
                p.add_mesh(original_stl, color='lightblue', opacity=0.3, label='Original STL')
                p.add_mesh(void_iso_surface, color='red', opacity=0.7, label='Original Voids')
                p.add_title("Before Alignment")
                p.add_legend()
                p.camera_position = 'iso'
                p.subplot(0, 1)
                p.add_mesh(final_stl, color='lightgreen', opacity=0.3, label='Final STL')
                p.add_mesh(final_voids, color='darkred', opacity=0.7, label='Final Voids')
                p.add_title("After Alignment")
                p.add_legend()
                p.camera_position = 'iso'
                p.show()
            patches = final_voids.split_bodies()
            if patches.n_blocks == 0:
                patch_list = [final_voids]
            else:
                cell_counts = [patch.n_cells for patch in patches]
                min_cells = 0.05 * max(cell_counts)
                patch_list = [patch for patch in patches if patch.n_cells > min_cells]
            decimated_patches = []
            for i, patch in enumerate(patch_list):
                decimated_patch = self.decimate_mesh(patch)
                decimated_patches.append(decimated_patch)
            if self.visualize:
                p = pv.Plotter()
                p.add_mesh(final_stl, color='lightblue', opacity=0.3, show_edges=True, label='STL Body')
                colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown']
                for i, patch in enumerate(decimated_patches):
                    color = colors[i % len(colors)]
                    p.add_mesh(patch, color=color, opacity=0.8, label=f'Void {i+1}')
                p.add_title("Final STL + Aligned Void Patches")
                p.add_legend()
                p.camera_position = 'iso'
                p.show()
            tm_base = self.to_trimesh(final_stl)
            if tm_base is None:
                print("Failed to convert STL to trimesh")
                return None
            trimesh.repair.fill_holes(tm_base)
            trimesh.repair.fix_normals(tm_base)
            result_tm = tm_base
            successful_subtractions = 0
            for i, patch in enumerate(decimated_patches):
                try:
                    tm_void = self.to_trimesh(patch)
                    if tm_void is None:
                        continue
                    print(f"Processing void patch {i+1}/{len(decimated_patches)}")
                    trimesh.repair.fill_holes(tm_void)
                    trimesh.repair.fix_normals(tm_void)
                    void_volume = tm_void.volume if tm_void.is_watertight else 0
                    stl_volume = result_tm.volume if result_tm.is_watertight else 0
                    if void_volume > 0 and stl_volume > 0:
                        volume_ratio = void_volume / stl_volume
                        if volume_ratio > 0.8:
                            print(f"Skipping void patch {i+1} due to large volume ratio: {volume_ratio:.2f}")
                            continue
                    if not tm_void.is_watertight:
                        print(f"Warning: Void patch {i+1} is not watertight, attempting to fix")
                        trimesh.repair.fill_holes(tm_void)
                        if not tm_void.is_watertight and len(tm_void.faces) < 1000:
                            tm_void = tm_void.convex_hull
                    print("Attempting boolean subtraction...")
                    new_result = trimesh.boolean.difference([result_tm, tm_void])
                    if new_result is None or len(new_result.faces) == 0:
                        print(f"Boolean subtraction failed for void patch {i+1}, skipping")
                        continue
                    if len(new_result.faces) < len(result_tm.faces) * 0.1:
                        print(f"Void patch {i+1} resulted in too few faces, skipping")  
                        continue
                    print(f"Void patch {i+1} successfully subtracted, new mesh has {len(new_result.faces)} faces")
                    result_tm = new_result
                    successful_subtractions += 1
                    if self.visualize:
                        p = pv.Plotter()
                        p.add_mesh(pv.PolyData(new_result.vertices, faces=np.hstack(
                            [np.full((len(new_result.faces), 1), 3), new_result.faces])),
                            color='orange', opacity=0.8, label='Boolean Result')
                        p.add_legend()
                        p.camera_position = 'iso'
                        p.show()
                except Exception as e:
                    print(f"Error processing void patch {i+1}: {e}")
                    continue
            if successful_subtractions == 0:
                return final_stl
            pv_result = pv.PolyData(result_tm.vertices, faces=np.hstack(
                [np.full((len(result_tm.faces), 1), 3), result_tm.faces]))
            cleaned = pv_result.clean(point_merging=True, merge_tol=1e-6)
            components = cleaned.split_bodies()
            if components.n_blocks > 1:
                component_sizes = [comp.n_cells for comp in components]
                max_size = max(component_sizes)
                large_components = [comp for comp in components if comp.n_cells >= max_size * 0.1]
                if len(large_components) == 1:
                    cleaned = large_components[0]
                else:
                    combined = large_components[0]
                    for comp in large_components[1:]:
                        combined = combined.merge(comp)
                    cleaned = combined
            if not isinstance(cleaned, pv.PolyData):
                if hasattr(cleaned, 'extract_surface'):
                    cleaned = cleaned.extract_surface()
                else:
                    cleaned = pv.PolyData(cleaned.points, faces=cleaned.faces)
            if hasattr(cleaned, 'is_all_triangles'):
                if not cleaned.is_all_triangles:
                    cleaned = cleaned.triangulate()
            else:
                cleaned = cleaned.triangulate()
            if cleaned.n_cells > 20000:
                cleaned = self.decimate_mesh(cleaned, target_ratio=0.6, max_triangles=20000)
            try:
                final_result = cleaned.compute_normals(
                    cell_normals=True,
                    point_normals=False,
                    consistent_normals=True,
                    auto_orient_normals=True
                )
            except Exception as norm_error:
                print(f"Warning: compute_normals failed: {norm_error}")
                final_result = cleaned
            if self.visualize:
                p = pv.Plotter(shape=(1, 2))
                p.subplot(0, 0)
                p.add_mesh(original_stl, color='lightblue', show_edges=True)
                p.add_title(f"Original STL\n({original_stl.n_cells} faces)")
                p.camera_position = 'iso'
                p.subplot(0, 1)
                p.add_mesh(final_result, color='lightgreen', show_edges=True)
                p.add_title(f"Final Result\n({final_result.n_cells} faces)")
                p.camera_position = 'iso'
                p.show()
            return final_result
        except Exception as e:
            print(f"Error in STL recovery process: {e}")
            traceback.print_exc()
            return None


if __name__ == "__main__":

    example = ExamplesCAD.Mitchell_1

    design_domain_stl_filepath, topopt_result_vtu_filepath, output_stl = get_example_cad(example)
    stlRecovery = TopOptSTLRecoveryWithAlignment(
        visualize=True,
        decimation_target_ratio=0.5,
        max_triangles_per_patch=10000,
        alignment_method='smart',
        scale_approach='scale_stl'
    )
    pv_result = stlRecovery.create_stl_from_topopt_result(
        design_domain_stl_filepath=design_domain_stl_filepath,
        topopt_result_vtu_filepath=topopt_result_vtu_filepath,
    )
    if pv_result is not None:
        result_tm = stlRecovery.to_trimesh(pv_result)
        if result_tm is not None:
            os.makedirs(os.path.dirname(output_stl), exist_ok=True)
            result_tm.export(output_stl)
        else:
            print("Failed to convert final result to trimesh")
    else:
        pass
