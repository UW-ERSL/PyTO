
import enum
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

class TrussOptExamples(enum.Enum):
    """
    Corresponding Truss optimization examples to the ones in the PyTO examples.
    """
    Mitchell_1 = enum.auto()
    EdgeCantilever = enum.auto()
    ShortCantileverTipLoad = enum.auto()
    ShortCantileverMidLoad = enum.auto()
    CantileverTipLoad = enum.auto()
    CantileverMidLoad = enum.auto()
    TwoBar = enum.auto()
    #ThreeHoleBracket = enum.auto()
    MBBB = enum.auto()
    DistributedLoad = enum.auto()
    Multiload = enum.auto()
    LBracketMidLoad = enum.auto()
    LBracketThick = enum.auto()

def get_trussopt_example(example: TrussOptExamples, width, height):
    """
    Get the truss optimization example based on the enum.
    """
    if example == TrussOptExamples.Mitchell_1:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,0] = 0
            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded

            #f += [0, -1] if (nd == [width, 0]).all() else [0, 0]
            f += [0, -1] if (nd[0] <= 0.1*width) and (nd[1] == 0) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.ShortCantileverTipLoad:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd[1] <= 0.1*height) and (nd[0] == width) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.ShortCantileverMidLoad:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
            # use below f we want to apply load at the middle on a line and not just at a point
            #f += [0, -1] if (nd[1] <= (height/2 + 0.1*height)) and (nd[1] >= height/2 - 0.1*height) and (nd[0] == width) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.CantileverTipLoad:
        #To Test width = 20, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd == [width, 0]).all() else [0, 0]
        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.CantileverMidLoad:
        #To Test width = 20, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.TwoBar:
        #To Test width = 20, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.MBBB:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,0] = 0
            if nd[0] >= 0.9*width and (nd[1] == 0): dof[i,1] = 0 # hard coded

            f += [0, -1] if (nd[0] <= 0.1*width) and (nd[1] == height) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.DistributedLoad:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0 and nd[1] == 0: dof[i,:] = [0, 0] 
            if nd[0] == width and nd[1] == 0: dof[i,:] = [0, 0] 

            f += [0, -1] if (nd[1] == height) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.Multiload:
        #To Test width = 10, height = 10
        poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), np.zeros((len(Nd), 2)), []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[0] == 0: dof[i,:] = [0, 0] 
            if (nd[0] == width) and (nd[1] == height/2):
                f[i,:] += [0, -1*0.1] 
            elif (nd[0] == width/2) and (nd[1] == height):
                f[i,:] += [0, -1] 
            else:
                f[i,:] = [0, 0]
        f = f.flatten().tolist()
        return poly, Nd, dof, f, PML, convex
    elif example == TrussOptExamples.LBracketMidLoad:
        #To Test width = 20, height = 10
        outer = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        # Subtract top-right corner to make L-shape
        cutout = Polygon([(width/2,height/2), (width,height/2), (width,height), (width/2,height)])

        # L-bracket domain
        poly = outer.difference(cutout)
        if (False): # set to True to create a hole in the middle of the domain
                poly = poly.difference(Polygon([
                    (width/4,     height/4),
                    (width/4*3,   height/4),
                    (width/4*3,   height/4*3),
                    (width/4,     height/4*3)
                ]))
        convex = True if poly.convex_hull.area == poly.area else False
        xv, yv = np.meshgrid(range(width+1), range(height+1))
        pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
        Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
        dof, f, PML = np.ones((len(Nd),2)), [], []
        #Load and support conditions
        for i, nd in enumerate(Nd):
            if nd[1] == height: dof[i,:] = [0, 0] 
            f += [0, -1] if (nd[1] <= 0.3*height) and (nd[1] >= 0.2*height) and (nd[0] >= width) else [0, 0]

        return poly, Nd, dof, f, PML, convex
    else:
        raise ValueError(f"Unknown example: {example}")
