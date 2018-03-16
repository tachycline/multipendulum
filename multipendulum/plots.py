# This is where the plotting functions go.
# for label formatting in latex
import sympy as sp
import numpy as np

# time series

# Phase space
def phase_plot(pend, ax, xcoord=None, ycoord=None):
    """Make a phase space plot in a provided set of axes.

    Parameters:
    -----------

    pend : a MultiPendulum instance
    ax : the set of axes to use for the plot
    xcoord : (optional) coordinate to use for the horizontal axis
    ycoord : (optional) coordinate to use for the vertical axis
    
    if xcoord and/or ycoord are not supplied, $\theta_0$ and $\dot{\theta}_0$
    will be used (respectively).

    Returns:
    --------
    Nothing.
    """
    if len(pend.timedf) == 0:
        pend.integrate_kane()
    
    if xcoord is None:
        xcoord = pend.q[0]
    if ycoord is None:
        ycoord = pend.u[0]
        
    titlestring = "Phase space plot"
    
    ax.plot(pend.timedf[xcoord], pend.timedf[ycoord])
    ax.set_xlabel(r"${}$".format(sp.latex(xcoord)), fontsize=18)
    ax.set_ylabel(r"${}$".format(sp.latex(ycoord)), fontsize=18)
    
# Power spectrum

# Animation

# Poincare plot
def poincare(pend, ax3d, coord=None):
    """Make a Poincare plot where coord==0.
    
    Parameters:
    -----------
    
    pend : a MultiPendulum instance
    ax3d : an Axes3D instance
    coord : (optional) which coordinate to use for the slicing critereon

    Returns:
    --------
    Nothing
    
    """
    if coord is None:
        coord = pend.q[0]
        
    signchange = (np.roll(np.sign(pend.timedf[coord]), 1)
                  - np.sign(pend.timedf[coord]) != 0).astype(int)
    
    section = pend.timedf[signchange.astype(bool)]
    
    coords = [c for c in pend.p + pend.q if c != coord]
    
    x = section[coords[0]].values
    y = section[coords[1]].values
    z = section[coords[2]].values
    
    ax3d.scatter(x, y, z)
    ax3d.set_xlabel(r"${}$".format(sp.latex(coords[0])))
    ax3d.set_ylabel(r"${}$".format(sp.latex(coords[1])))
    ax3d.set_zlabel(r"${}$".format(sp.latex(coords[2])))
    