##########################################################################
# This Python script was written by L. He, M. Gilbert, X. Song           #
# University of Sheffield, United Kingdom                                #
# Please send comments to: linwei.he@sheffield.ac.uk                     #
# The script is intended for educational purposes - theoretical details  #
# are discussed in the following paper, which should be cited in any     #
# derivative works or technical papers which use the script:             #
#                                                                        #
# "A Python script for adaptive layout optimization of trusses",         #
# L. He, M. Gilbert, X. Song, Struct. Multidisc. Optim., 2018            #
#                                                                        #
# Disclaimer:                                                            #
# The authors reserve all rights but do not guarantee that the script is #
# free from errors. Furthermore, the authors are not liable for any      #
# issues caused by the use of the program.                               #
##########################################################################

# Script is further modified to export the truss design to a CSV file to be used in PyTO.
import sys
sys.path.append('../PyTO-1/src') #assuming the PyTO is in the parent directory
from math import gcd, ceil
import itertools
from scipy import sparse
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from truss_opt_examples import *
from topopt_benchmarks import StructuralTOExamples

#Calculate equilibrium matrix B
def calcB(Nd, Cn, dof):
    m, n1, n2 = len(Cn), Cn[:,0].astype(int), Cn[:,1].astype(int)
    l, X, Y = Cn[:,2], Nd[n2,0]-Nd[n1,0], Nd[n2,1]-Nd[n1,1]
    d0, d1, d2, d3 = dof[n1*2], dof[n1*2+1], dof[n2*2], dof[n2*2+1]
    s = np.concatenate((-X/l * d0, -Y/l * d1, X/l * d2, Y/l * d3))
    r = np.concatenate((n1*2, n1*2+1, n2*2, n2*2+1))
    c = np.concatenate((np.arange(m), np.arange(m), np.arange(m), np.arange(m)))
    return sparse.coo_matrix((s, (r, c)), shape = (len(Nd)*2, m))
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
    q = [np.array(qi.value).flatten() for qi in q]
    a = np.array(a.value).flatten()
    u = [-np.array(eqnk.dual_value).flatten() for eqnk in eqn]
    return vol, a, q, u
#Check dual violation
def stopViolation(Nd, PML, dof, st, sc, u, jc):
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
#Visualize truss
def plotTruss(Nd, Cn, a, q, threshold, str, update = True):
    plt.ion() if update else plt.ioff()
    plt.clf(); plt.axis('off'); plt.axis('equal');  plt.draw()
    plt.title(str)
    tk = 5 / max(a)
    for i in [i for i in range(len(a)) if a[i] >= threshold]:
        if all([q[lc][i]>=0 for lc in range(len(q))]): c = 'r'
        elif all([q[lc][i]<=0 for lc in range(len(q))]): c = 'b'
        else: c = 'tab:gray'
        pos = Nd[Cn[i, [0, 1]].astype(int), :]
        plt.plot(pos[:, 0], pos[:, 1], c, linewidth = a[i] * tk)
    plt.pause(0.01) if update else plt.show()

# Plot shapely polygon
def plot_shapely_domain(domain: Polygon, Nd, dof, f):
    # Interpret DOF and forces
    Nd = np.array(Nd)
    dof = np.array(dof).reshape((-1, 2))
    f = np.array(f).reshape((-1, 2))

    # Extract support types
    fully_fixed = Nd[np.all(dof == 0, axis=1)]
    x_only_fixed = Nd[(dof[:, 0] == 0) & (dof[:, 1] == 1)]
    y_only_fixed = Nd[(dof[:, 0] == 1) & (dof[:, 1] == 0)]

    # Extract loads
    loads = Nd[np.any(f != 0, axis=1)]

    # Plot the polygon
    fig, ax = plt.subplots(figsize=(10, 5))
    x, y = domain.exterior.xy
    ax.plot(x, y, 'k', linewidth=2, label='Domain')

    for interior in domain.interiors:
        ix, iy = interior.xy
        ax.plot(ix, iy, 'k')

    # Plot all nodes
    ax.plot(Nd[:, 0], Nd[:, 1], 'o', markersize=3, color='lightgray', label='Nodes')

    # Plot supports
    if len(fully_fixed):
        ax.plot(fully_fixed[:, 0], fully_fixed[:, 1], 's', color='blue', label='Fully Fixed (x,y)')
    if len(x_only_fixed):
        ax.plot(x_only_fixed[:, 0], x_only_fixed[:, 1], 'o', color='green', label='X Fixed')
    if len(y_only_fixed):
        ax.plot(y_only_fixed[:, 0], y_only_fixed[:, 1], 'o', color='purple', label='Y Fixed')

    # Plot loads
    if len(loads):
        ax.plot(loads[:, 0], loads[:, 1], 'v', color='red', label='Loads')

    ax.set_aspect('equal')
    ax.set_title("Boundary Conditions: Supports and Loads")
    ax.legend()
    plt.grid(True)
    plt.show()


def plot_truss_with_domain_and_bcs(domain: Polygon, Nd, dof, f, Cn, a, q, threshold=1e-4, title="Truss and Domain"):
    import matplotlib.pyplot as plt

    Nd = np.array(Nd)
    Cn = np.array(Cn)
    a = np.array(a)
    dof = np.array(dof).reshape((-1, 2))
    f = np.array(f).reshape((-1, 2))

    # Extract support types
    fully_fixed = Nd[np.all(dof == 0, axis=1)]
    x_only_fixed = Nd[(dof[:, 0] == 0) & (dof[:, 1] == 1)]
    y_only_fixed = Nd[(dof[:, 0] == 1) & (dof[:, 1] == 0)]

    # Extract loads
    loads = Nd[np.any(f != 0, axis=1)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Plot domain
    x, y = domain.exterior.xy
    ax.plot(x, y, 'k', linewidth=2, label='Domain')
    for interior in domain.interiors:
        ix, iy = interior.xy
        ax.plot(ix, iy, 'k')

    # --- Plot all nodes
    ax.plot(Nd[:, 0], Nd[:, 1], 'o', markersize=3, color='lightgray', label='Nodes')

    # --- Supports
    if len(fully_fixed):
        ax.plot(fully_fixed[:, 0], fully_fixed[:, 1], 's', color='blue', label='Fully Fixed (x,y)')
    if len(x_only_fixed):
        ax.plot(x_only_fixed[:, 0], x_only_fixed[:, 1], 'o', color='green', label='X Fixed')
    if len(y_only_fixed):
        ax.plot(y_only_fixed[:, 0], y_only_fixed[:, 1], 'o', color='purple', label='Y Fixed')

    # --- Loads
    if len(loads):
        ax.plot(loads[:, 0], loads[:, 1], 'v', color='red', label='Loads')

    # --- Plot truss bars
    tk = 5 / max(a)
    for i in range(len(a)):
        if a[i] < threshold:
            continue
        n1, n2 = Cn[i, 0:2].astype(int)
        pos = Nd[[n1, n2], :]
        if all([qk[i] >= 0 for qk in q]):
            color = 'r'
        elif all([qk[i] <= 0 for qk in q]):
            color = 'b'
        else:
            color = 'gray'
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=a[i]*tk)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.show()

#Main function 
def trussopt(trussopt_problem, truss_width, truss_height, st = 1, sc = 1, jc = 1, b_plot: bool = False):
    """
    Truss optimization function.
    Args:
        trussopt_problem: Truss optimization problem to solve.
        st: Tension limit.
        sc: Compression limit.
        jc: Joint length limit.  #larger jc value if you want only a handful of members in the final design.
        b_plot: Boolean flag to plot the truss design.;
    Returns:
        Nd: Node coordinates - list of nodes (each row = [x,y])
        Cn: Connectivity matrix - 
            [ start_node_index, end_node_index, length, is_active (0 or 1) ]
        a: Member areas - optimal area for each row in Cn.
        q: Member forces - optimal force for each row in Cn. q>0 means tension, q<0 means compression.
    """
    # poly = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
    # if (False): # set to True to create a hole in the middle of the domain
    #         poly = poly.difference(Polygon([
    #             (width/4,     height/4),
    #             (width/4*3,   height/4),
    #             (width/4*3,   height/4*3),
    #             (width/4,     height/4*3)
    #         ]))
    # convex = True if poly.convex_hull.area == poly.area else False
    # xv, yv = np.meshgrid(range(width+1), range(height+1))
    # pts = [Point(xv.flat[i], yv.flat[i]) for i in range(xv.size)]
    # Nd = np.array([[pt.x, pt.y] for pt in pts if poly.intersects(pt)])
    # dof, f, PML = np.ones((len(Nd),2)), [], []
    # #Load and support conditions
    # for i, nd in enumerate(Nd):
    #     if nd[0] == 0: dof[i,:] = [0, 0] 
    #     f += [0, -1] if (nd == [width, height/2]).all() else [0, 0]
    poly, Nd, dof, f, PML, convex = get_trussopt_example(trussopt_problem, truss_width, truss_height)
    if b_plot:
        plot_shapely_domain(poly, Nd, dof, f)
    
    #Create the 'ground structure'
    for i, j in itertools.combinations(range(len(Nd)), 2):
        dx, dy = abs(Nd[i][0] - Nd[j][0]), abs(Nd[i][1] - Nd[j][1])
        if gcd(int(dx), int(dy)) == 1 or jc != 0:
            seg = [] if convex else LineString([Nd[i], Nd[j]])
            if convex or poly.contains(seg) or poly.boundary.contains(seg):
                PML.append( [i, j, np.sqrt(dx**2 + dy**2), False] )
    PML, dof = np.array(PML), np.array(dof).flatten()
    f = [f[i:i+len(Nd)*2] for i in range(0, len(f), len(Nd)*2)]
    print('Nodes: %d Members: %d' % (len(Nd), len(PML)))
    for pm in [p for p in PML if p[2] <= 1.42]: #1.42 is length of the member (hard coded, change if needed)
        pm[3] = True
    #Start the 'member adding' loop
    for itr in range(1, 100):
        Cn = PML[PML[:,3] == True]
        vol, a, q, u = solveLP(Nd, Cn, f, dof, st, sc, jc)
        print("Itr: %d, vol: %f, mems: %d" % (itr, vol, len(Cn)))
        #plotTruss(Nd, Cn, a, q, max(a) * 1e-3, "Itr:" + str(itr))
        if stopViolation(Nd, PML, dof, st, sc, u, jc): break
    print("Volume: %f" % (vol))
    if b_plot: 
        plotTruss(Nd, Cn, a, q, max(a) * 1e-3, "Finished", False)
        plot_truss_with_domain_and_bcs(poly, Nd, dof, f, Cn, a, q, threshold=1e-4, title="Truss and Domain")
    return Nd, Cn, a, q

#Execution function when called directly by Python
if __name__ =='__main__': 
    trussopt_problem = TrussOptExamples.Mitchell_1
    width, height = 10, 10
    trussopt(trussopt_problem, width, height, b_plot = True) #much larger jc value if you want only a handful of members in the final design.
