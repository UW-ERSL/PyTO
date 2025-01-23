import math
from stl import mesh  # pip install numpy-stl
import vtk  # pip install vtk
from collections import defaultdict
from queue import Queue
from tkinter import filedialog

class STLGeom:
    TOL = 1e-9  # Tolerance for normalization

    def __init__(self, file_path):
        self.mesh = mesh.Mesh.from_file(file_path)
        self.stl_n_triangles = len(self.mesh.vectors)

        # Compute triangle properties
        self.tri_normals = [self.compute_normal(vertices) for vertices in self.mesh.vectors]
        self.tri_areas = [self.get_area_of_triangle(i) for i in range(self.stl_n_triangles)]
        self.tri_neighbors = self.compute_neighbors()

        # Initialize highlight states and selection
        self.tri_highlight = [False] * self.stl_n_triangles  # Highlight states
        self.selected_triangles = set()  # User-selected triangles

        # File path for visualization
        self.file_path = file_path

    def compute_normal(self, vertices):
        v1 = [vertices[1][i] - vertices[0][i] for i in range(3)]
        v2 = [vertices[2][i] - vertices[0][i] for i in range(3)]

        normal = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        norm = math.sqrt(sum(n ** 2 for n in normal)) or self.TOL
        return [n / norm for n in normal]

    def compute_neighbors(self):
        edge_map = defaultdict(list)  # Map of edges to triangle indices
        neighbors = [[] for _ in range(self.stl_n_triangles)]

        for i, vertices in enumerate(self.mesh.vectors):
            edges = [
                tuple(sorted((tuple(vertices[0]), tuple(vertices[1])))),
                tuple(sorted((tuple(vertices[1]), tuple(vertices[2])))),
                tuple(sorted((tuple(vertices[2]), tuple(vertices[0])))),
            ]

            for edge in edges:
                edge_map[edge].append(i)

        for edge, tri_list in edge_map.items():
            for t1 in tri_list:
                for t2 in tri_list:
                    if t1 != t2 and t2 not in neighbors[t1]:
                        neighbors[t1].append(t2)

        return neighbors

    def get_area_of_triangle(self, triangle_index):
        vertices = self.mesh.vectors[triangle_index]
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        return self.compute_area_of_triangle(x, y, z)

    def compute_area_of_triangle(self, x, y, z):
        v1 = [x[1] - x[0], y[1] - y[0], z[1] - z[0]]
        v2 = [x[2] - x[0], y[2] - y[0], z[2] - z[0]]

        cross_product = [
            v1[1] * v2[2] - v2[1] * v1[2],
            -(v1[0] * v2[2] - v2[0] * v1[2]),
            v1[0] * v2[1] - v2[0] * v1[1],
        ]
        cross_product_norm = math.sqrt(sum(c ** 2 for c in cross_product))
        return 0.5 * cross_product_norm
    
    def highlight_triangles_recursive(self, seed_triangle, depth, cutoff_angle_degrees):
        """
        Toggle highlight state for triangles recursively based on the angle between normals.
        If the seed triangle is highlighted, recursively deselect; otherwise, highlight.
        """
        cumulative_area = 0
        cos_theta = math.cos(math.radians(cutoff_angle_degrees))

        # Determine the target state (toggle behavior)
        target_state = not self.tri_highlight[seed_triangle]

        # Initialize queue
        q = Queue()
        q.put((seed_triangle, depth))
        self.tri_highlight[seed_triangle] = target_state

        while not q.empty():
            current_tri, current_depth = q.get()
            if current_depth == 0:
                continue

            cumulative_area += self.get_area_of_triangle(current_tri)
            n1 = self.tri_normals[current_tri]

            for neighbor_tri in self.tri_neighbors[current_tri]:
                n2 = self.tri_normals[neighbor_tri]
                if self.tri_highlight[neighbor_tri] != target_state and self.dot_product(n1, n2) > cos_theta:
                    self.tri_highlight[neighbor_tri] = target_state
                    q.put((neighbor_tri, current_depth - 1))

        return cumulative_area
    
    @staticmethod
    def dot_product(v1, v2):
        """
        Compute the dot product of two 3D vectors.
        """
        return sum(a * b for a, b in zip(v1, v2))

    def visualize_with_vtk(self):
        # Create vtkPolyData to hold the STL geometry
        poly_data = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        # Add points and triangles to vtkPolyData
        for vertices in self.mesh.vectors:
            point_ids = []
            for v in vertices:
                point_ids.append(points.InsertNextPoint(v))
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, point_ids[0])
            triangle.GetPointIds().SetId(1, point_ids[1])
            triangle.GetPointIds().SetId(2, point_ids[2])
            cells.InsertNextCell(triangle)

        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)

        # Mapper and actor for STL geometry
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Mapper and actor for highlighted triangles
        highlight_poly_data = vtk.vtkPolyData()
        highlight_mapper = vtk.vtkPolyDataMapper()
        highlight_mapper.SetInputData(highlight_poly_data)

        highlight_actor = vtk.vtkActor()
        highlight_actor.SetMapper(highlight_mapper)
        highlight_actor.GetProperty().SetColor(1, 0, 0)  # Red for highlights
        highlight_actor.GetProperty().SetOpacity(0.6)  # Semi-transparent

        # Renderer and render window
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.AddActor(highlight_actor)
        renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background
        renderer.ResetCamera()

        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(800, 600)

        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Add coordinate system
        axes = vtk.vtkAxesActor()
        axes_widget = vtk.vtkOrientationMarkerWidget()
        axes_widget.SetOrientationMarker(axes)
        axes_widget.SetInteractor(render_window_interactor)
        axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        axes_widget.SetEnabled(1)
        axes_widget.InteractiveOn()

        # Interaction style
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        render_window_interactor.SetInteractorStyle(interactor_style)

        # Picker for triangle selection
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)

        def update_highlights():
            """
            Update the highlighted triangles in the visualization.
            """
            highlight_points = vtk.vtkPoints()
            highlight_cells = vtk.vtkCellArray()

            for i, highlight in enumerate(self.tri_highlight):
                if highlight or i in self.selected_triangles:
                    vertices = self.mesh.vectors[i]
                    point_ids = []
                    for v in vertices:
                        point_ids.append(highlight_points.InsertNextPoint(v))
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, point_ids[0])
                    triangle.GetPointIds().SetId(1, point_ids[1])
                    triangle.GetPointIds().SetId(2, point_ids[2])
                    highlight_cells.InsertNextCell(triangle)

            highlight_poly_data.SetPoints(highlight_points)
            highlight_poly_data.SetPolys(highlight_cells)
            render_window.Render()

        def on_left_button_press(obj, event):
            """
            Handle left mouse button press for selecting/deselecting triangles.
            Select all triangles in the same plane recursively.
            """
            click_pos = render_window_interactor.GetEventPosition()
            picker.Pick(click_pos[0], click_pos[1], 0, renderer)

            cell_id = picker.GetCellId()
            if cell_id >= 0:
                # Recursively highlight triangles in the same plane
                self.highlight_triangles_recursive(seed_triangle=cell_id, depth=10, cutoff_angle_degrees=90)
                update_highlights()

        def rotate_callback(obj, event):
            actor.RotateX(1)
            actor.RotateY(1)
            update_highlights()
            render_window.Render()

        def zoom_callback(obj, event):
            renderer.GetActiveCamera().Zoom(1.2)
            update_highlights()
            render_window.Render()

        def pan_callback(obj, event):
            renderer.GetActiveCamera().Azimuth(5)
            renderer.GetActiveCamera().Elevation(5)
            renderer.ResetCameraClippingRange()
            update_highlights()
            render_window.Render()

        # Add observers
        render_window_interactor.AddObserver("LeftButtonPressEvent", on_left_button_press)
        render_window_interactor.AddObserver("MiddleButtonPressEvent", zoom_callback)
        render_window_interactor.AddObserver("RightButtonPressEvent", pan_callback)

        # Start interaction
        update_highlights()
        render_window.Render()
        render_window_interactor.Start()


# Example usage
if __name__ == "__main__":
    stl_file = filedialog.askopenfilename(
        title="Select STL file",
        filetypes=[("STL files", "*.stl *.STL")]
    )
    if not stl_file:
        print("No file selected")
        exit()
    
    stl_geom = STLGeom(stl_file)

    # Pre-highlight triangles using recursive logic
    seed_triangle = 0
    cutoff_angle = 90  # Degrees
    stl_geom.highlight_triangles_recursive(seed_triangle, depth=5, cutoff_angle_degrees=cutoff_angle)

    # Visualize the STL with enhanced interaction
    stl_geom.visualize_with_vtk()
