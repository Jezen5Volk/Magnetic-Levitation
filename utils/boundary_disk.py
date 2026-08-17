from scipy.optimize import curve_fit
from scipy.integrate import solve_bvp
import numpy as np



class Cylinder:
    def __init__(self, 
            b,  # radius in meters
            rho, # cylinder density in kg/m^3
            Omega = 1000, # angular velocity in rpm 
            rho_f = 1.226, # density of the fluid in kg/m^3
            mu = 1.795e-5, # viscosity of the fluid in pascal seconds
            elle = 0.0105, # total gap distance in meters
            L = 0.003175 # length of the cylinder in meters
        ):

        self.b = b
        self.rho = rho
        self.Omega = Omega * (2*np.pi/60) #convert from rpm to rads/second
        self.rho_f = rho_f
        self.mu = mu
        self.elle = elle
        self.L = L

        return


    '''
    Curve Fitting + Experimental Results
    '''
    def linear_damping(self, gap, tau, start, stop):
        '''
        Under an assumption of uniform linear damping, fit the ring down times as a function of gap 
        distance using a logarithmic approach. Use the result to estimate the dissipation in 
        watts/m^2

        gap: a series of measured gap distances in meters [m]
        tau: a corresponding series of measured exponential ring down times in seconds [s]
        start: the start point for the theoretical curve for the dissipation in meters [m]
        stop: the stop point for the theoretical curve for the dissipation in meters [m]
        '''

        #curve fit tau
        s_array = np.linspace(start, stop, 100)
        popt, _ = curve_fit(lambda x, a, b: a*np.log(x) + b, gap, tau, p0 = (1, 1))
        tau_fit = popt[0]*np.log(s_array) + popt[1]

        #dissipation
        dissipation = (np.pi**2 *self.rho* self.b**3)/(3600*tau)*self.Omega**2
        dissipation_fit = (np.pi**2 *self.rho* self.b**3)/(3600*tau_fit)*self.Omega**2

        return s_array, dissipation, dissipation_fit



    def enclosed_rotor_stator(self, gap, tau, start, stop):
        '''
        As a first approach to nonlinear fitting, use the results of Daily and Nece (1960) to 
        model the disk face of the cylinder as it approaches the wall. This function assumes the
        shaft of the cylinder remains in the laminar regime, estimating the dissipation in watts/m^2
        
        
        gap: a series of measured gap distances in meters [m]
        tau: a corresponding series of measured exponential ring down times in seconds [s]
        start: the start point for the theoretical curve for the dissipation in meters [m]
        stop: the stop point for the theoretical curve for the dissipation in meters [m]
        '''

        # TODO: implement a means by which to incorporate experimental data into this plot

        
        s_array = np.linspace(start, stop, 100)
        s_crit = (1.62*self.b)/(self.rho_f*self.Omega*self.b**2/self.mu)**(5/11) 

        #Three components of cylindrical dissipation
        D_disk_1 = np.where(
            s_array < s_crit,
            (self.mu * self.b**2 * self.Omega**2) / (s_array),
            (1.85 * np.sqrt(self.rho_f * self.mu) * self.b**(19/10)) / np.pi * (s_array)**0.1 * self.Omega**2.5
        )
        D_shaft = 2 * self.mu * self.b * self.Omega**2
        D_disk_2 = (1.85 * np.sqrt(self.rho_f * self.mu) * self.b**(19/10)) / np.pi * (self.elle - self.L - s_array)**0.1 * self.Omega**2.5 

        dissipation = D_disk_1 + D_disk_2 + D_shaft

        return s_array, dissipation, s_crit



