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


    
    def open_rotor_stator(self, gap, tau, start, stop):
           '''
           As a second approach to nonlinear fitting, modify the results of Von Karman (1921) to 
           model the disk face of the cylinder as it approaches the wall. This function assumes the
           shaft of the cylinder remains in the laminar regime, estimating the dissipation in watts/m^2
           
           
           gap: a series of measured gap distances in meters [m]
           tau: a corresponding series of measured exponential ring down times in seconds [s]
           start: the start point for the theoretical curve for the dissipation in meters [m]
           stop: the stop point for the theoretical curve for the dissipation in meters [m]
           '''
   
           # TODO: implement a means by which to incorporate experimental data into this plot
   
           
           s_array, C_m = self.disk_moment_coefficient(start, stop)

           D_disk_1 = (C_m*0.5*self.rho_f*self.Omega**2*self.b**5)/(np.pi**2 *self.b)
           
           D_shaft = 2 * self.mu * self.b * self.Omega**2
   
           dissipation = D_disk_1 + D_shaft
   
           return s_array, dissipation
   


    '''
    Moment Coefficient Numerical ODE Fitting
    '''
    def disk_moment_coefficient(self, start, stop):
        '''
        Compute G'(0) for the moment coefficient of a rotating disk
        '''

        #sweep distance
        s_array = np.linspace(start, stop, 100)

        s_sound = []
        G_prime = []
        for s in s_array:
            #setup mesh
            s_1 = s*np.sqrt(self.Omega/self.mu)
            z = np.linspace(0, s_1, 100)

            #initial guess
            y_guess = np.zeros((5, z.size))
            y_guess[0] = 0.2 * z * np.exp(-z) # F guess
            y_guess[2] = np.exp(-z)            # G guess
            y_guess[4] = -0.5 * z              # H guess
            p_guess = [s]

            sol = solve_bvp(disk_wall_ode, disk_wall_bc, z, y_guess, p_guess)

            if sol.success:
                s_sound.append(s)
                G_prime.append(sol.sol(z)[3][0])

        G_prime = np.asarray(G_prime)
        R = self.Omega*self.b**2*self.rho_f/self.mu
        C_m = -np.pi*G_prime*R**0.5


        return np.asarray(s_sound), C_m



'''
ODE Equations, separate from Class
'''
def disk_wall_ode(z, y, p):
    K = p[0]
    F, F_prime, G, G_prime, H = y
    
    
    dF = F_prime
    dF_prime = F**2 - G**2 + F_prime * H + K**2
    dG = G_prime
    dG_prime = 2 * F * G + G_prime * H
    dH = -2 * F
    
    return np.vstack((dF, dF_prime, dG, dG_prime, dH))


def disk_wall_bc(ya, yb, p):
    '''
    ya corresponds to z_1 = 0
    yb corresponds to z_1 = s sqrt(omega/nu)
    p is for unfixed constants, but is unused in the boundary conditions

    the scipy BVP function is setup to return residuals
    '''

    return np.array([
        ya[0] - 0.0,  # F(0) = 0
        ya[2] - 1.0,  # G(0) = 1
        ya[4] - 0.0,  # H(0) = 0
        yb[0] - 0.0,  # F(z(s)) = 0
        yb[2] - 0.0,  # G(z(s)) = 0
        yb[4] - 0.0   # H((s))
    ])
