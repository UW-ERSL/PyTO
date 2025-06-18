import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
import time
from math import gcd, ceil
import itertools
from scipy import sparse
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import enum
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

from topopt_benchmarks import *
from truss_3D_opt_examples import *
from truss_domain_using_pyvista import *
#Calculate equilibrium matrix B
def calcB(Nd, Cn, dof):
    m = len(Cn)
    n1 = Cn[:, 0].astype(int)
    n2 = Cn[:, 1].astype(int)
    dx = Nd[n2, 0] - Nd[n1, 0]
    dy = Nd[n2, 1] - Nd[n1, 1]
    dz = Nd[n2, 2] - Nd[n1, 2]
    l = Cn[:, 2]  # member length

    # Get the directional components
    ux = dx / l
    uy = dy / l
    uz = dz / l

    # Indices into the flattened DOF vector
    d0 = dof[n1 * 3]
    d1 = dof[n1 * 3 + 1]
    d2 = dof[n1 * 3 + 2]
    d3 = dof[n2 * 3]
    d4 = dof[n2 * 3 + 1]
    d5 = dof[n2 * 3 + 2]

    # Fill the matrix
    s = np.concatenate((
        -ux * d0,
        -uy * d1,
        -uz * d2,
         ux * d3,
         uy * d4,
         uz * d5
    ))
    r = np.concatenate((
        n1 * 3,
        n1 * 3 + 1,
        n1 * 3 + 2,
        n2 * 3,
        n2 * 3 + 1,
        n2 * 3 + 2
    ))
    c = np.concatenate([np.arange(m)] * 6)

    return sparse.coo_matrix((s, (r, c)), shape=(len(Nd) * 3, m))

#Solve linear programming problem
def solveLP(Nd, Cn, f, dof, st, sc, jc):
    l = [col[2] + jc for col in Cn]
    B = calcB(Nd, Cn, dof)
    a = cvx.Variable(len(Cn))
    obj = cvx.Minimize(np.transpose(l) * a)
    q, eqn, cons= [],  [], [a>=0]
    for k, fk in enumerate(f):
        q.append(cvx.Variable(len(Cn)))
        eqn.append(B * q[k] == fk * dof)
        cons.extend([eqn[k], q[k] >= -sc * a, q[k] <= st * a])
    prob = cvx.Problem(obj, cons)
    vol = prob.solve()
    print("Solver status:", prob.status)
    q = [np.array(qi.value).flatten() for qi in q]
    a = np.array(a.value).flatten()
    u = [-np.array(eqnk.dual_value).flatten() for eqnk in eqn]
    return vol, a, q, u
#Check dual violation
def stopViolation(poly, Nd, PML, dof, st, sc, u, jc):
    lst = np.where(PML[:,3]==False)[0]
    Cn = PML[lst]
    l = Cn[:,2] + jc
    B = calcB(Nd, Cn, dof).tocsc()
    y = np.zeros(len(Cn))
    for uk in u:
        yk = np.multiply(B.transpose().dot(uk) / l, np.array([[st], [-sc]]))
        y += np.amax(yk, axis=0)
    vioCn = np.where(y>1.0001)[0]
    vioSort = np.flipud(np.argsort(y[vioCn]))
    num = ceil(min(len(vioSort), 0.05*max( [len(Cn)*0.05, len(vioSort)])))
    for i in range(num): 
        PML[lst[vioCn[vioSort[i]]]][3] = True
    return num == 0

def visualize_grid_and_selected(domain, xv, yv, zv, Nd):
    # Full grid points
    grid_points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
    grid = pv.PolyData(grid_points)

    # Selected Nd points
    Nd_pts = pv.PolyData(Nd)

    p = pv.Plotter()
    p.add_mesh(domain, style='wireframe', color='lightgray', opacity=0.3)
    p.add_points(grid, color='gray', point_size=6, render_points_as_spheres=True, label='Full grid')
    p.add_mesh(domain, style='wireframe', color='lightgray', opacity=0.3)
    p.add_points(Nd_pts, color='red', point_size=10, render_points_as_spheres=True, label='Inside Nd')

    p.add_legend()
    p.show()

def compute_voxel_diagonal(Nd):
    """
    Compute the body diagonal of a single voxel from Nd (node positions).
    Assumes structured grid, returns the diagonal length of a single cube/voxel.
    """
    Nd = np.array(Nd)
    diffs = []
    for axis in range(3):
        sorted_axis = np.unique(Nd[:, axis])
        if len(sorted_axis) > 1:
            spacing = np.min(np.diff(sorted_axis))
        else:
            spacing = 0
        diffs.append(spacing)
    
    dx, dy, dz = diffs
    return np.sqrt(dx**2 + dy**2 + dz**2) + 1e-4  # add small epsilon
#Main function 
def trussopt3D(truss_example: Truss3DOptExamples, poly, Nd, dof, f, st, sc, jc, b_plot=False):
    
    convex = True if is_polygon_convex(poly) else False
    PML = []
    #Create the 'ground structure'
    for i, j in itertools.combinations(range(len(Nd)), 2):
        dx, dy, dz = abs(Nd[i][0] - Nd[j][0]), abs(Nd[i][1] - Nd[j][1]), abs(Nd[i][2] - Nd[j][2])
        if gcd(int(dx), int(dy), int(dz)) == 1 or jc != 0:
            seg = [] if convex else pv.Line(tuple(map(float, Nd[i])), tuple(map(float, Nd[j])))
            
            if convex or is_segment_inside_domain(poly, seg):
                PML.append( [i, j, np.sqrt(dx**2 + dy**2 + dz**2), False] )
    PML, dof = np.array(PML), np.array(dof).flatten()
    f = [f[i:i+len(Nd)*3] for i in range(0, len(f), len(Nd)*3)]
    print('Nodes: %d Members: %d' % (len(Nd), len(PML)))
    threshold_length = compute_voxel_diagonal(Nd)
    ls_sel_mem = [p for p in PML if p[2] <= threshold_length]
    #ls_sel_mem = [p for p in PML if p[2] <= 1.75]
    for pm in ls_sel_mem: 
        pm[3] = True
    
    Cn = PML[PML[:,3] == True]
    if b_plot:
        plot_truss_with_domain_and_bcs(poly, Nd, dof, f, Cn, threshold=1e-4, title="Initial Truss")    
    #Start the 'member adding' loop
    for itr in range(1, 100):
        Cn = PML[PML[:,3] == True]
        vol, a, q, u = solveLP(Nd, Cn, f, dof, st, sc, jc)
        print("Itr: %d, vol: %f, mems: %d" % (itr, vol, len(Cn)))
        if b_plot:
            plot_truss_with_domain_and_bcs(poly, Nd, dof, f, Cn, a, q, threshold=1e-4, title="Itr:" + str(itr))
        if stopViolation(poly, Nd, PML, dof, st, sc, u, jc): break
    print("Volume: %f" % (vol)) 
    return Nd, Cn, a, q


def get_truss_width_height(mesh):
    """
    Get the width and height of the truss from the mesh.
    
    Parameters:
    - mesh: hex_mesher.HexMesher object
    
    Returns:
    minimum value of 10 for the smallest dimension so that the truss grid can have unit spacing.
    - truss_width: width of the truss
    - truss_height: height of the truss
    - truss_depth: depth of the truss
    """
    # Determine physical size of mesh
    mesh_width = mesh.bbox.x.max - mesh.bbox.x.min
    mesh_height = mesh.bbox.y.max - mesh.bbox.y.min
    mesh_depth = mesh.bbox.z.max - mesh.bbox.z.min

    # Determine scale factor so that smallest dimension becomes 10
    scale_factor = 10.0 / min(mesh_width, mesh_height, mesh_depth)

    # Compute desired truss domain size
    truss_width = int(np.ceil(mesh_width * scale_factor))
    truss_height = int(np.ceil(mesh_height * scale_factor))
    truss_depth = int(np.ceil(mesh_depth * scale_factor))
    return truss_width, truss_height, truss_depth

#Execution function when called directly by Python
if __name__ =='__main__': 
    
    starttime = time.time()
    print("-" * 50)
    to_problem = StructuralTOExamples.LBracketMidLoad # Choose the TO problem
    print(f"Running {to_problem.name}...") 
    print("-" * 50)

    # Get the structural problem
    mesh, mat_prop, bc,elem_body_force, to_params = getStructuralTOProblem(to_problem)
    truss_width, truss_height, truss_depth  = get_truss_width_height(mesh)
    
    truss_example, poly, Nd, dof, f = get_trussopt_3D_example(to_problem, truss_width, truss_height, truss_depth)
    print('open edges', poly.n_open_edges)
    plot_3d_domain_with_bconditions(poly, Nd, dof, f)
    Nd, Cn, a, q = trussopt3D(truss_example, poly, Nd, dof, f, st = 1, sc =1, jc = 0.1, b_plot=False)
    endtime = time.time()
    plot_truss_with_domain_and_bcs(poly, Nd, dof, f, Cn, a, q, threshold=1e-4, title="Truss and Domain", update=False)
    print(f"Execution time: {endtime - starttime:.2f} seconds")
##########################################################################
# A Python script for 3D truss optimization.                             #  
# It is a modified version of the 2D script mentioned below.             #
# It uses PyVista for domain creation and visualization,                 #
# and CVXPY for solving the optimization problem.                        #   
# The original Python script was taken from                              #   
# "A Python script for adaptive layout optimization of trusses",         #
# L. He, M. Gilbert, X. Song, Struct. Multidisc. Optim., 2018            #   
# The script is intended for educational purposes - theoretical details  #
# are discussed in the following paper, which should be cited in any     #
# derivative works or technical papers which use the script:             #                                                                     #
# Disclaimer:                                                            #
# The authors reserve all rights but do not guarantee that the script is #
# free from errors. Furthermore, the authors are not liable for any      #
# issues caused by the use of the program.                               #
##########################################################################