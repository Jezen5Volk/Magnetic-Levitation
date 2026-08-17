from scipy.optimize import curve_fit
from scipy.integrate import solve_bvp
import numpy as np



class Cylinder:
    def __init__(self, 
            b = 0.00254,  # radius in meters
            rho = 7500, # cylinder density in kg/m^3
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
        gap_array = np.linspace(start, stop, 100)
        popt, _ = curve_fit(lambda x, a, b: a*np.log(x) + b, gap, tau, p0 = (1, 1))
        tau_fit = popt[0]*np.log(gap_array) + popt[1]

        #dissipation
        dissipation = (np.pi**2 *self.rho* self.b**3)/(3600*tau)*self.Omega**2
        dissipation_fit = (np.pi**2 *self.rho* self.b**3)/(3600*tau_fit)*self.Omega**2

        return dissipation, tau_fit, dissipation_fit


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

        
        gap_analytic = np.linspace(start, stop, 100)





def von_karman_ode(z, y, p):
    K = p[0]
    F, F_prime, G, G_prime, H = y
    
    
    dF = F_prime
    dF_prime = F**2 - G**2 + F_prime * H + K**2
    dG = G_prime
    dG_prime = 2 * F * G + G_prime * H
    dH = -2 * F
    
    return np.vstack((dF, dF_prime, dG, dG_prime, dH))

# boundary conditions
def bc(ya, yb, p):
    # ya corresponds to z_1 = 0
    # yb corresponds to z_1 = s sqrt(omega/nu)
    # p is for unfixed constants, but is unused

    #return residuals
    return np.array([
        ya[0] - 0.0,  # F(0) = 0
        ya[2] - 1.0,  # G(0) = 1
        ya[4] - 0.0,  # H(0) = 0
        yb[0] - 0.0,  # F(z(s)) = 0
        yb[2] - 0.0,  # G(z(s)) = 0
        yb[4] - 0.0   # H((s))
    ])

# Setup mesh/
Omega = 1000 #angular velocity in rpm 
mu = 1.795e-5 #viscosity of the fluid
s = 0.003*np.sqrt(Omega*2*np.pi/(60*mu))
z = np.linspace(0, s, 100)

# Initial guess for state variables
y_guess = np.zeros((5, z.size))
y_guess[0] = 0.2 * z * np.exp(-z) # F guess
y_guess[2] = np.exp(-z)            # G guess
y_guess[4] = -0.5 * z              # H guess

p_guess = [s]

# 4. Solve BVP
sol = solve_bvp(von_karman_ode, bc, z, y_guess, p_guess)

if sol.success:
    print("BVP solved successfully!")
    
    # 5. Plot the solution
    z_plot = np.linspace(0, s, 300)
    y_plot = sol.sol(z_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(z_plot, y_plot[0], label=r'$F(z_1)$ (radial)')
    plt.plot(z_plot, y_plot[2], label=r'$G(z_1)$ (tangential)')
    plt.plot(z_plot, -y_plot[4], label=r'$H(z_1)$ (axial)')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.xlim(0, s)
    plt.xlabel(r'$z_1$')
    plt.ylabel('Dimensionless Velocity Components')
    plt.title('von Kármán Swirling Flow Profiles')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("BVP solver failed to converge:", sol.message)