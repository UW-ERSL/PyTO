
import enum
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

from truss_domain_using_pyvista import *
from topopt_benchmarks import StructuralTOExamples

class Truss3DOptExamples(enum.Enum):
    """
    Corresponding Truss optimization examples to the ones in the PyTO examples.
    """
    
    EdgeSupportedCantilever = enum.auto()
    Mitchell_1 = enum.auto()
    # EdgeCantilever = enum.auto()
    ShortCantileverTipLoad = enum.auto()
    ShortCantileverMidLoad = enum.auto()
    CantileverTipLoad = enum.auto()
    CantileverMidLoad = enum.auto()
    TwoBar = enum.auto()
    # #ThreeHoleBracket = enum.auto()
    MBBB = enum.auto()
    # DistributedLoad = enum.auto()
    # Multiload = enum.auto()
    # LBracketMidLoad = enum.auto()
    # LBracketThick = enum.auto()
    LBracket = enum.auto()


def get_trussopt_3D_example(to_problem: StructuralTOExamples, width, height, depth, scaling = 10.0):
    if to_problem == StructuralTOExamples.EdgeCantilever:
        truss_example = Truss3DOptExamples.EdgeSupportedCantilever
        
        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            #if np.isclose(nd[0], 0): dof[i,:] = [0, 0, 0] 
            if (nd[0] == 0 and nd[2] == 0) or (nd[0] == 0 and nd[2] == depth): dof[i,:] = [0, 0, 0] # fix side edges of the x=0 fae 
            if np.allclose(nd, [width, height / 2, depth / 2], atol=1e-3):
                f += [0, -1, 0]
            else:
                f += [0, 0, 0]
    elif to_problem == StructuralTOExamples.CantileverMidLoad:
        truss_example = Truss3DOptExamples.CantileverMidLoad

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            #if np.isclose(nd[0], 0): dof[i,:] = [0, 0, 0] 
            if nd[0] == 0: dof[i,:] = [0, 0, 0] # fix entire x=0 face
            f += [0, -1, 0] if (nd == [width, height/2, depth/2]).all() else [0, 0, 0]
    elif to_problem == StructuralTOExamples.Mitchell_1:
        truss_example = Truss3DOptExamples.Mitchell_1

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,0] = 0
            if nd[0] == 0: dof[i,2] = 0
            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded
            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,2] = 0 # hard coded

            #f += [0, -1] if (nd == [width, 0]).all() else [0, 0]
            f += [0, -1, 0] if (nd[0] <= 0.1*width) and (nd[1] == 0) else [0, 0, 0]
    elif to_problem == StructuralTOExamples.ShortCantileverTipLoad:
        truss_example = Truss3DOptExamples.ShortCantileverTipLoad

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0, 0] 
            f += [0, -1, 0] if (nd[1] <= 0.1*height) and (nd[0] == width) else [0, 0, 0]
    elif to_problem == StructuralTOExamples.ShortCantileverMidLoad:
        truss_example = Truss3DOptExamples.ShortCantileverMidLoad

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0, 0] 
            f += [0, -1, 0] if (nd == [width, height/2, depth/2]).all() else [0, 0, 0]
    elif to_problem == StructuralTOExamples.CantileverTipLoad:
        truss_example = Truss3DOptExamples.CantileverTipLoad

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0, 0] 
            f += [0, -1, 0] if (nd[1] <= 0.1*height) and (nd[0] == width) else [0, 0, 0]
    elif to_problem == StructuralTOExamples.TwoBar:
        truss_example = Truss3DOptExamples.TwoBar

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0, 0] 
            f += [0, -1, 0] if (nd == [width, height/2., depth/2.]).all() else [0, 0, 0]
    elif to_problem == StructuralTOExamples.MBBB:
        truss_example = Truss3DOptExamples.MBBB

        poly = create_domain_with_optional_cutout(width, height, depth, make_hole=False)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        spacing_x = width / 6   
        spacing_y = height / 4  
        spacing_z = depth / 2  

        x = np.arange(0, nx, spacing_x)
        y = np.arange(0, ny, spacing_y)
        z = np.arange(0, nz, spacing_z)
        # Convert to int explicitly
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,0] = 0
            if nd[0] == 0: dof[i,2] = 0

            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded
            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,2] = 0 # hard coded
            
            f += [0, -1, 0] if (nd[0] <= 0.1*width) and (nd[1] == height) else [0, 0, 0]
    elif to_problem == StructuralTOExamples.LBracketMidLoad:
        truss_example = Truss3DOptExamples.LBracket
        #width, height, depth = 3, 4, 1
        cut_width_ratio = 0.6
        cut_height_ratio = 0.6
        poly = make_lbracket(width, height, cut_width=height*cut_height_ratio, cut_height=width*cut_width_ratio, depth=depth)
        bounds = poly.bounds
        Lx = bounds[1] - bounds[0]
        Ly = bounds[3] - bounds[2]
        Lz = bounds[5] - bounds[4]
        nx, ny, nz = int(Lx) + 1, int(Ly) + 1, int(Lz) + 1

        cut_width = width * cut_width_ratio
        cut_height = height * cut_height_ratio

        
        spacing_x = width / 4   
        spacing_y = height / 4  
        spacing_z = depth / 2 
        # Compute number of intervals per region
        nx1 = int(np.ceil(cut_width / spacing_x))        # x: cutout region
        nx2 = int(np.ceil((width - cut_width) / spacing_x))
        ny1 = int(np.ceil(cut_height / spacing_y))       # y: cutout region
        ny2 = int(np.ceil((height - cut_height) / spacing_y))
        nz = int(np.ceil(depth / spacing_z))

        # Define grid points for each region
        x1 = np.linspace(0, cut_width, nx1, endpoint=False)
        x2 = np.linspace(cut_width, width, nx2 + 1)
        y1 = np.linspace(0, cut_height, ny1, endpoint=False)
        y2 = np.linspace(cut_height, height, ny2 + 1)
        z = np.linspace(0, depth, nz + 1)

        # Vertical leg: x ∈ [0, cut_width], y ∈ [0, height]
        xv1, yv1, zv1 = np.meshgrid(x1, np.concatenate([y1, y2]), z, indexing='ij')
        points1 = np.stack([xv1.ravel(), yv1.ravel(), zv1.ravel()], axis=1)

        # Horizontal leg: x ∈ [cut_width, width], y ∈ [0, cut_height]
        xv2, yv2, zv2 = np.meshgrid(x2, y1, z, indexing='ij')
        points2 = np.stack([xv2.ravel(), yv2.ravel(), zv2.ravel()], axis=1)

        # Combine and return
        grid_points = np.vstack([points1, points2])
        Nd = np.array([pt for pt in grid_points if is_point_inside_domain(poly, pt)])
        #visualize_grid_and_selected(poly, xv, yv, zv, Nd)
        
        #Load and support conditions
        dof, f = np.ones((len(Nd),3)), []
        for i, nd in enumerate(Nd):
            if nd[0] == height: dof[i,:] = [0, 0, 0] #LBracket
            f += [-1, 0, 0] if (nd[1] == width and nd[0] == height*(1-cut_height_ratio)//2.).all() else [0, 0, 0]
    else:
        raise ValueError("Unknown truss problem type.")
    return truss_example, poly, Nd, dof, f

# def get_trussopt_3D_example(example: Truss3DOptExamples, width, height):
#     """
#     Get the truss optimization example based on the enum.
#     """
#     if example == Truss3DOptExamples.Mitchell_1:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,0] = 0
#             if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded

#             #f += [0, -1] if (nd == [width, 0]).all() else [0, 0]
#             f += [0, -1] if (nd[0] <= 0.1*width) and (nd[1] == 0) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.ShortCantileverTipLoad:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd[1] <= 0.1*height) and (nd[0] == width) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.ShortCantileverMidLoad:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
#             # use below f we want to apply load at the middle on a line and not just at a point
#             #f += [0, -1] if (nd[1] <= (height/2 + 0.1*height)) and (nd[1] >= height/2 - 0.1*height) and (nd[0] == width) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.CantileverTipLoad:
#         #To Test width = 20, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd == [width, 0]).all() else [0, 0]
#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.CantileverMidLoad:
#         #To Test width = 20, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.TwoBar:
#         #To Test width = 20, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.MBBB:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,0] = 0
#             if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded

#             f += [0, -1] if (nd[0] <= 0.1*width) and (nd[1] == height) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.DistributedLoad:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0 and nd[1] == 0: dof[i,:] = [0, 0] 
#             if nd[0] == width and nd[1] == 0: dof[i,:] = [0, 0] 

#             f += [0, -1] if (nd[1] == height) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.Multiload:
#         #To Test width = 10, height = 10
#         poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), np.zeros((len(Nd), 2)), []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[0] == 0: dof[i,:] = [0, 0] 
#             if (nd[0] == width) and (nd[1] == height/2):
#                 f[i,:] += [0, -1*0.1] 
#             elif (nd[0] == width/2) and (nd[1] == height):
#                 f[i,:] += [0, -1] 
#             else:
#                 f[i,:] = [0, 0]
#         f = f.flatten().tolist()
#         return poly, Nd, dof, f, PML, convex
#     elif example == Truss3DOptExamples.LBracketMidLoad:
#         #To Test width = 20, height = 10
#         outer = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
#         # Subtract top-right corner to make L-shape
#         cutout = Polygon([(width/2,height/2), (width,height/2), (width,height), (width/2,height)])

#         # L-bracket domain
#         poly = outer.difference(cutout)
#         if (False): # set to True to create a hole in the middle of the domain
#                 poly = poly.difference(Polygon([
#                     (width/4,     height/4),
#                     (width/4*3,   height/4),
#                     (width/4*3,   height/4*3),
#                     (width/4,     height/4*3)
#                 ]))
#         convex = True if poly.convex_hull.area == poly.area else False
#         xv, yv = np.meshgrid(range(width+1), range(height+1))
#         pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
#         Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
#         dof, f, PML = np.ones((len(Nd),2)), [], []
#         #Load and support conditions
#         for i, nd in enumerate(Nd):
#             if nd[1] == height: dof[i,:] = [0, 0] 
#             f += [0, -1] if (nd[1] <= 0.3*height) and (nd[1] >= 0.2*height) and (nd[0] >= width) else [0, 0]

#         return poly, Nd, dof, f, PML, convex
#     else:
#         raise ValueError(f"Unknown example: {example}")
