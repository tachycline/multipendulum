# coding: utf-8
# %load multipendulum/multipendulum.py
import numpy as np
import sympy as sp
import pandas as pd
import h5py

from sympy import symbols
from sympy.physics import mechanics

from sympy import Dummy, lambdify
from scipy.integrate import odeint

from numpy.fft import rfft, rfftfreq

import matplotlib
if matplotlib.get_backend() == 'Qt5Agg':
    matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

class MultiPendulum(object):
    """Class for simulating a multiple pendulum system."""

    def __init__(self, n):
        """Initialize a multi-pendulum with n links."""

        self.n = n
        self.timeseries = None
        
        # coordinates
        self.q = mechanics.dynamicsymbols('theta_:{0}'.format(n))
        self.qdot = mechanics.dynamicsymbols('theta_:{0}'.format(n), 1)
        self.u = mechanics.dynamicsymbols('thetadot_:{0}'.format(n))
        self.p = mechanics.dynamicsymbols('p_:{}'.format(n))

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
        self.origin = P

        # lists to hold particles, forces, and kinetic ODEs
        # for each pendulum in the chain
        particles = []
        forces = []
        kinetic_odes = []
        
        # energy bits
        self.energies = dict()
        self.energies["T"] = 0
        self.energies["V"] = 0


        for i in range(n):
            # Create a reference frame following the i^th mass
            Ai = A.orientnew('A_' + str(i), 'Axis', [self.q[i], A.x])
            Ai.set_ang_vel(A, self.u[i] * A.x)

            # Create a point in this reference frame
            Pi = P.locatenew('P_' + str(i), -self.l[i] * Ai.z)
            Pi.v2pt_theory(P, A, Ai)

            # Create a new particle of mass m[i] at this point
            Pai = mechanics.Particle('Pa_' + str(i), Pi, self.m[i])
            particles.append(Pai)

            # Set forces & compute kinematic ODE
            forces.append((Pi, -self.m[i] * self.g * A.z))
            kinetic_odes.append(self.q[i].diff(self.t) - self.u[i])
            
            # build energy terms
            Ti = Pai.kinetic_energy(A)
            self.energies["T_{}".format(i)] = Ti
            self.energies["T"] += Ti
            
            posvec = mechanics.express(Pi.pos_from(self.origin), A)
            Vi = posvec.dot(A.z)*self.m[i]*self.g
            self.energies["V_{}".format(i)] = Vi
            self.energies["V"] += Vi

            P = Pi
            
        # Lagrangian, total energy
        self.energies["E"] = self.energies["T"] + self.energies["V"]
        self.energies["L"] = self.energies["T"] - self.energies["V"]
        
        # canonical momenta
        self.pdef = dict()
        for i in range(n):
            self.pdef[self.p[i]] = sp.diff(self.energies["L"], self.u[i])
            
        self.psubs = sp.solve([sp.Eq(p[0], p[1]) for p in self.pdef.items()], self.u)
        
        # substitute for Hamiltonian
        self.energies['H'] = self.energies["E"].subs(self.psubs)
        
        # for external use with energetics -- maybe temporary?
        self.A = A
        self.particles = particles
        self.forces = forces
        self.kinetic_odes = kinetic_odes

        # calculate eigenmodes/eigenfrequencies
        # self.calculate_linear_eigenmodes()
        self.build_energy_func()
        
        # default times for integration
        self.times = np.linspace(0,100,10000)

    def set_initial_conditions(self, theta_0, omega_0, degrees=False, eigenmodes=False):
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

    def build_energy_func(self):
        """Build functions to calculate energy terms and total energy

        Produces three dictionaries:
        * self.energies, which contains purely symbolic forms for the energy terms
        * self.numerical_energies: as above, but with numerical values for l and m
        * self.efuncs: executable functions for calculating energy terms for a point
                       in phase space
        """

        params = [self.g] + list(self.l) + list(self.m)
        param_vals = [9.81] + list(self.lengths) + list(self.masses)

        # substitute in parameters to get numerical expressions
        self.numerical_energies = dict()
        for label, energy in self.energies.items():
            self.numerical_energies[label] = energy.subs(dict(zip(params,
                                                                  param_vals)))
        # finally lambdify to make executable functions
        self.efuncs = dict()

        for label, energy in self.numerical_energies.items():
            if label == 'H':
                coords = list(self.q) + list(self.p)
            else:
                coords = list(self.q) + list(self.u)
            self.efuncs[label] = sp.lambdify(coords, energy)

    def make_energy_timeseries(self, offsets=True):
        """Applies the energy functions to the results of an integration

        Stores the results in a dictionary, keyed by the same labels used in
        self.energies, etc.

        If offsets is True, the origin in phase space is set to zero energy.
        If offsets is False, the pivot point is set to zero potential energy.
        """

        offset = 0.0
        for label, energy in self.efuncs.items():
            if offsets:
                offset = energy(0,0,0,0)

            if label == 'H':
                columns = [self.timedf[coord] for coord in self.q + self.p]
            else:
                columns = [self.timedf[coord] for coord in self.q + self.u]
                
            self.timedf[label] = energy(*columns) - offset

    def perturb(self, magnitude=1.0e-10, direction=None):
        coords = list(self.q) + list(self.u)
                
        grad_E = np.array([sp.lambdify(coords, 
                                       sp.diff(self.numerical_energies['E'], coord))(*self.y0) 
                           for coord in coords])
        grad_E_normalized = grad_E/np.sqrt(grad_E.dot(grad_E))
        if direction is None:
            direction = np.random.random(4)
            
        a = direction - direction.dot(grad_E_normalized)*grad_E_normalized
        ahat = a/np.sqrt(a.dot(a))
        perturbation = ahat*magnitude
        self.y0 += perturbation

    def calculate_linear_eigenmodes(self):
        """Calculates linear eigenmodes via diagonalization"""

        op_point = dict(zip(self.q+self.u, np.zeros_like(self.q+self.u)))
        A, B, C = self.KM.linearize(op_point=op_point, A_and_B=True, new_method=True)

        # pull out the quadrant of the matrix we care about
        Asimp = -sp.simplify(A)[self.n:2*self.n, 0:self.n]

        # substitute in numerical values for parameters
        parameters = [self.g] + list(self.l) + list(self.m)
        parameter_vals = [9.81] + list(self.lengths) + list(self.masses)
        Anumerical = Asimp.subs(dict(zip(parameters, parameter_vals)))
        self.S, self.D = Anumerical.diagonalize()

    def integrate_hamiltonian(self, times=None, rtol=1.5e-8, atol=1.5e-8):
        """Carry out the integration using Hamilton's equations."""
        
        if times is None:
            times = self.times
        else:
            self.times = times
            
        coords = list(self.q) + list(self.p)
            
        rhs = dict()
        # build Hamilton's equations
        for i in range(self.n):
            rhs[self.q[i]] = sp.lambdify(coords, sp.diff(self.numerical_energies['H'], self.p[i]))
            rhs[self.p[i]] = sp.lambdify(coords, -sp.diff(self.numerical_energies['H'], self.q[i]))
            
        def gradient(y,t):
            rvals = np.zeros_like(y)
            for idx, coord in enumerate(coords):
                rvals[idx] = rhs[coord](*y)
            return rvals
        
        # perform the integration
        self.timeseries = odeint(gradient, self.y0, times, rtol=rtol, atol=atol)
        
        # make a dataframe and find the qdots
        tsdict = dict()
        for idx in range(self.n):
            tsdict[self.q[idx]] = self.timeseries[:,idx]
            tsdict[self.p[idx]] = self.timeseries[:,self.n+idx]
            
        self.timedf = pd.DataFrame(tsdict, index=pd.Index(self.times))

        constants = list(self.l) + list(self.m)
        constant_values = list(self.lengths) + list(self.masses)
        for vel in self.u:
            velfunc = sp.lambdify((self.q + self.p), self.psubs[vel].subs(dict(zip(constants, constant_values))))

            columns = [self.timedf[coord] for coord in self.q + self.p]
            self.timedf[vel] = velfunc(*columns)

    def integrate_kane(self, times=None, rtol=1.5e-8, atol=1.5e-8):
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
            
        # Generate Kane's equations of motion
        self.KM = mechanics.KanesMethod(self.A, q_ind=self.q, u_ind=self.u,
                                   kd_eqs=self.kinetic_odes)
        self.fr, self.fr_star = self.KM.kanes_equations(self.particles, self.forces)

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
        self.timeseries = odeint(gradient, self.y0, times, args=(parameter_vals,), rtol=rtol, atol=atol)

        # make a dataframe and find the p_i
        tsdict = dict()
        for idx in range(self.n):
            tsdict[self.q[idx]] = self.timeseries[:,idx]
            tsdict[self.u[idx]] = self.timeseries[:,self.n+idx]
            
        self.timedf = pd.DataFrame(tsdict, index=pd.Index(self.times))

        constants = list(self.l) + list(self.m)
        constant_values = list(self.lengths) + list(self.masses)
        for p in self.p:
            momfunc = sp.lambdify((self.q + self.u), self.pdef[p].subs(dict(zip(constants, constant_values))))

            columns = [self.timedf[coord] for coord in self.q + self.u]
            self.timedf[p] = momfunc(*columns)
        
        
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
        x, y = get_xy_coords(self.timeseries, self.lengths)

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        radius = sum(self.lengths)*1.1
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius))

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
        if self.n > 1:
            for idx, axis in enumerate(ax):
                axis.plot(timeseries[:,idx], timeseries[:,self.n+idx])
                axis.set_xlabel(r"${}$".format(sp.latex(coordinates[idx])), fontsize=18)
                axis.set_ylabel(r"${}$".format(sp.latex(velocities[idx])), fontsize=18)
                axis.set_title(titlestring.format(idx), fontsize=22)
        else:
            ax.plot(timeseries[:,0], timeseries[:,1])
            ax.set_xlabel(r"${}$".format(sp.latex(coordinates[0])), fontsize=18)
            ax.set_ylabel(r"${}$".format(sp.latex(velocities[0])), fontsize=18)
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
        """Compute and plot a power spectrum"""

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
            ax.text(freq, ylim[0], "{:.2f}".format(freq), color="red", va="bottom", ha="center",
                   bbox=dict(facecolor='white', edgecolor='white'))
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
