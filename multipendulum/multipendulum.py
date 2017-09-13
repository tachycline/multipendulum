# coding: utf-8
import numpy as np
import sympy as sp
import h5py

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint

from numpy.fft import rfft, rfftfreq

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class MultiPendulum(object):
    """Class for simulating a multiple pendulum system."""
    
    def __init__(self, n):
        """Initialize a multi-pendulum with n links."""
        
        self.n = n
        self.timeseries = None
        self.q = mechanics.dynamicsymbols('theta_:{0}'.format(n))
        self.u = mechanics.dynamicsymbols('thetadot_:{0}'.format(n))

        # mass and length
        self.m = symbols('m_:{0}'.format(n))
        self.l = symbols('l_:{0}'.format(n))

        # gravity and time symbols
        self.g, self.t = symbols('g,t')
        
        # default values for mass and length
        self.lengths = np.broadcast_to(1/n, n)
        self.masses = np.broadcast_to(1.0, n)
        
        
        # Create pivot point reference frame
        A = mechanics.ReferenceFrame('A')
        P = mechanics.Point('P')
        P.set_vel(A, 0)

        # lists to hold particles, forces, and kinetic ODEs
        # for each pendulum in the chain
        particles = []
        forces = []
        kinetic_odes = []

        for i in range(n):
            # Create a reference frame following the i^th mass
            Ai = A.orientnew('A_' + str(i), 'Axis', [self.q[i], A.z])
            Ai.set_ang_vel(A, self.u[i] * A.z)

            # Create a point in this reference frame
            Pi = P.locatenew('P_' + str(i), self.l[i] * Ai.x)
            Pi.v2pt_theory(P, A, Ai)

            # Create a new particle of mass m[i] at this point
            Pai = mechanics.Particle('Pa_' + str(i), Pi, self.m[i])
            particles.append(Pai)

            # Set forces & compute kinematic ODE
            forces.append((Pi, self.m[i] * self.g * A.x))
            kinetic_odes.append(self.q[i].diff(self.t) - self.u[i])

            P = Pi

        # Generate equations of motion
        self.KM = mechanics.KanesMethod(A, q_ind=self.q, u_ind=self.u,
                                   kd_eqs=kinetic_odes)
        self.fr, self.fr_star = self.KM.kanes_equations(forces, particles)
        
        # calculate eigenmodes/eigenfrequencies
        self.calculate_linear_eigenmodes()
        
        # default times for integration
        self.times = np.linspace(0,100,10000)
        
    def set_initial_conditions(self, theta_0, omega_0, degrees=True, eigenmodes=False):
        """Set initial conditions.
        
        Parameters
        ----------
        theta_0: float or iterable of floats
            contains inital position amplitudes. Should either be a single value or
            a tuple, list, or array of n floats, where n is the number of links.
            
        omega_0: float or iterable of floats
            contains initial velocity amplitudes. Should either be a single value or
            a tuple, list, or array of n floats, where n is the number of links.
            
        If theta_0 and/or omega_0 contain a single value, that value will be broadcast
        to all positions/velocities.
        
        degrees: Boolean
            If true, interpret theta_0 and omega_0 as degrees, and convert to radians.
            
        eigenmodes: Boolean
            If true, theta_0 and omega_0 are interpreted as amplitudes for the linear
            eigenmodes instead of individual positons/velocities.
            
            
        Returns:
        --------
        Nothing. (Initial conditions are stored in self.y0)
        """
        if degrees:
            y0 = np.deg2rad(np.concatenate([np.broadcast_to(theta_0, self.n),
                                                 np.broadcast_to(omega_0, self.n)]))
        else:
            y0 = np.concatenate([np.broadcast_to(theta_0, self.n),
                                      np.broadcast_to(omega_0, self.n)])

        if eigenmodes:
            positions = (self.S * sp.Matrix([y0[0:self.n]]).T).T
            velocities = (self.S * sp.Matrix([y0[self.n:2*self.n]]).T).T
            self.y0 = np.array(positions.tolist()[0] + velocities.tolist()[0]).astype(np.float64)
            
        else:
            self.y0 = y0
            
    
    def set_lengths(self, lengths):
        """setter for lengths"""
        self.lengths = np.broadcast_to(lengths, self.n)
        # recalculate eigenmodes
        self.calculate_linear_eigenmodes()
    
    def set_masses(self, masses):
        """setter for masses"""
        self.masses = np.broadcast_to(masses, self.n)
        # recalculate eigenmodes
        self.calculate_linear_eigenmodes()
    
    def calculate_linear_eigenmodes(self):
        op_point = dict(zip(self.q+self.u, np.zeros_like(self.q+self.u)))
        A, B, C = self.KM.linearize(op_point=op_point, A_and_B=True, new_method=True)
        Asimp = -sp.simplify(A)[self.n:2*self.n, 0:self.n]
        
        parameters = [self.g] + list(self.l) + list(self.m)
        parameter_vals = [9.81] + list(self.lengths) + list(self.masses)
        Anumerical = Asimp.subs(dict(zip(parameters, parameter_vals)))
        self.S, self.D = Anumerical.diagonalize()
                
            
    def integrate(self, times=None):
        """Carry out the integration.
        
        Parameters:
        -----------
        times: numpy array of time values for integration; optional.
            if not supplied, the integrator will use whatever is stored
            in self.times, which by default goes from 0 to 100 in steps of 0.01.
            
        Returns:
        --------
        Nothing, but stores output in self.timeseries.
        """
        
        # deal with times argument
        if times is None:
            times = self.times
        else:
            self.times = times
        
        # Fixed parameters: gravitational constant, lengths, and masses
        parameters = [self.g] + list(self.l) + list(self.m)
        parameter_vals = [9.81] + list(self.lengths) + list(self.masses)

        # define symbols for unknown parameters
        unknowns = [Dummy() for i in self.q + self.u]
        unknown_dict = dict(zip(self.q + self.u, unknowns))
        kds = self.KM.kindiffdict()

        # substitute unknown symbols for qdot terms
        mm_sym = self.KM.mass_matrix_full.subs(kds).subs(unknown_dict)
        fo_sym = self.KM.forcing_full.subs(kds).subs(unknown_dict)

        # create functions for numerical calculation 
        mm_func = lambdify(unknowns + parameters, mm_sym)
        fo_func = lambdify(unknowns + parameters, fo_sym)

        # function which computes the derivatives of parameters
        def gradient(y, t, args):
            vals = np.concatenate((y, args))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            return np.array(sol).T[0]

        # ODE integration
        self.timeseries = odeint(gradient, self.y0, times, args=(parameter_vals,))
    
    def project_timeseries_to_eigenmodes(self):
        """Project the timeseries onto the eigenmode basis."""
        
        thetats = self.timeseries[:,0:self.n].T
        thetadotts = self.timeseries[:,self.n:2*self.n].T

        eigts = np.matmul(np.array(self.S.inv()).astype(np.float64), thetats)
        eigvts = np.matmul(np.array(self.S.inv()).astype(np.float64), thetadotts)

        self.eigts = np.vstack((eigts, eigvts)).T
    
    
    def animate(self, times=None):
        """Generate an animation"""
        
        self.integrate(times)
        x, y = get_xy_coords(self.timeseries)
    
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

        line, = ax.plot([], [], 'o-', lw=2)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(x[i], y[i])
            return line,

        self.anim = animation.FuncAnimation(fig, animate, frames=len(self.times),
                                            interval=1000 * self.times.max() / len(self.times),
                                            blit=True, init_func=init)
        plt.close(fig)
    
    def phase_plots(self, eigenmodes=False):
        """Make phase space plots.
        
        Parameters
        ----------
        
        eigenmodes: Boolean
            if true, phase plots are of eigenmodes. If false, phase plots are of
            individual links in the chain of pendulums.
            
        Returns:
        --------
        The figure instance.
        """
        if self.timeseries is None:
            raise ValueError("Nothing to plot!")
        
        if eigenmodes:
            coordinates = sp.symbols("phi_:{}".format(self.n))
            velocities = sp.symbols("phidot_:{}".format(self.n))
            self.project_timeseries_to_eigenmodes()
            timeseries = self.eigts
            titlestring = "Phase space plot, linear eigenmode {}"
        else:
            coordinates = self.q
            velocities = self.u
            timeseries = self.timeseries
            titlestring = "Phase space plot, pendulum {}"
        
        
        fig, ax = plt.subplots(ncols=self.n)
        fig.set_figwidth(self.n*10)
        fig.set_figheight(10)
        for idx, axis in enumerate(ax):
            axis.plot(timeseries[:,idx], timeseries[:,self.n+idx])
            axis.set_xlabel(r"${}$".format(sp.latex(coordinates[idx])), fontsize=18)
            axis.set_ylabel(r"${}$".format(sp.latex(velocities[idx])), fontsize=18)
            axis.set_title(titlestring.format(idx), fontsize=22)
        #plt.close(fig)
        return fig
            
    def time_series_plots(self, eigenmodes=False):
        """Plot time series.
        
        Parameters
        ----------
        eigenmodes: Boolean
            if true, plot eigenmode amplitudes as a function of time. If false,
            plot the amplitudes of individual coordinates.
            
        Returns the figure.        
        """
        fig,ax = plt.subplots(nrows=4, ncols=1, sharex=True)
        fig.set_figwidth(16)
        fig.set_figheight(16)
        coordinates = list(self.q) + list(self.u)

        if eigenmodes:
            self.project_timeseries_to_eigenmodes()
            for i in range(self.n):
                ax[i].plot(self.times, self.eigts[:,i])
                ax[i].set_ylabel(r"$\phi_{}$".format(i), fontsize=16)
                ax[self.n+i].plot(self.times, self.eigts[:,self.n+i])
                ax[self.n+i].set_ylabel(r"$\dot\phi_{}$".format(i), fontsize=16)
            ax[0].set_title("Eigenmode timeseries", fontsize=24)
        else:
            for i in range(self.n*2):
                ax[i].plot(self.times, self.timeseries[:,i])
                ax[i].set_ylabel(r"${}$".format(sp.latex(coordinates[i])), fontsize=16)
    
            ax[0].set_title("Coordinate timeseries", fontsize=24)
        
        ax[-1].set_xlabel("time (s)", fontsize=16)
        return fig
        
    def serialize(self, filename="MultiPendulum.h5"):
        """Write the integration results to an HDF5 archive.
        
        Parameters:
        -----------
        filename: string, optional
            gives the name of the HDF5 archive.
        """
        
        runkey = tuple(list(self.lengths) + list(self.masses) + list(self.y0))
        timekey = (np.min(self.times), np.max(self.times), len(self.times))
        
        path = "{}/{}/{}".format(self.n, runkey, timekey)
        
        outfile = h5py.File(filename, 'a')
        try:
            mygrp = outfile[path]
        except KeyError:
            mygrp = outfile.create_group(path)
        
        try:
            mygrp['times'] = self.times
            mygrp['timeseries'] = self.timeseries
            mygrp['eigts'] = self.eigts
        except RuntimeError: # already present; don't save again.
            pass
        outfile.close()
        
    def powerspectrum(self, eigenmodes=False):
        spacing = (np.max(self.times) - np.min(self.times))/len(self.times)
        frequencies = rfftfreq(len(self.times), spacing)


        markerfreqs = [sp.sqrt(self.D[i,i])/(np.pi*2) for i in range(self.n)]

        fig, ax = plt.subplots()
        fig.set_figwidth(12)
        fig.set_figheight(6)
        
        if eigenmodes:
            pseries = self.eigts
            labels = [r"\phi_{}".format(i) for i in range(self.n)]
            title = "Frequency Power Spectrum, eigenmode basis"
        else:
            pseries = self.timeseries
            labels = [sp.latex(label) for label in self.q]
            title = "Frequency Power Spectrum, coordinate basis"
            
        for i in range(self.n):
            amplitude = rfft(pseries[:,i])
            power = np.abs(amplitude)**2
            ax.plot(frequencies, power, label="${}$".format(labels[i]))
        
        ax.set_ylabel("Power")
        ax.set_xlabel("Frequency ($s^{-1}$)")
        ax.loglog()
        ax.set_title(title)
        ylim = ax.get_ylim()
        for freq in markerfreqs:
            line, = ax.plot([freq, freq], ylim, ":r")
        line.set_label("Linear eigenmode frequencies")
        ax.set_ylim(ylim)
        ax.legend()
        return fig
        

def get_xy_coords(p, lengths=None):
    """Get (x, y) coordinates from generalized coordinates p"""
    p = np.atleast_2d(p)
    n = p.shape[1] // 2
    if lengths is None:
        lengths = np.ones(n) / n
    zeros = np.zeros(p.shape[0])[:, None]
    x = np.hstack([zeros, lengths * np.sin(p[:, :n])])
    y = np.hstack([zeros, -lengths * np.cos(p[:, :n])])
    return np.cumsum(x, 1), np.cumsum(y, 1)
