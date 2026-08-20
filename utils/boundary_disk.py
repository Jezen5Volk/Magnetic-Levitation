from scipy.optimize import curve_fit
from scipy.integrate import solve_bvp
from scipy.special import lambertw
import numpy as np



class Cylinder:
    def __init__(self, 
            b,  # radius in meters
            rho, # cylinder density in kg/m^3
            Omega, # angular velocity in rpm 
            rho_f, # density of the fluid in kg/m^3
            mu, # viscosity of the fluid in pascal seconds
            elle, # total gap distance in meters
            L # length of the cylinder in meters
        ):

        self.b = b
        self.rho = rho
        self.Omega = Omega * (2*np.pi/60) #convert from rpm to rads/second
        self.rho_f = rho_f
        self.mu = mu
        self.elle = elle
        self.L = L
        self.R = (self.rho_f*self.Omega*self.b**2)/self.mu #rotational reynold's number
        self.M = self.rho*(np.pi*self.b**2*self.L) #mass of cylinder
        self.I = 0.5 *self.M*self.b**2 #rotational moment of inertia

        #Viscous torque for the cylindrical shaft
        if self.R < 60:
            self.M_shaft = (4*self.L*self.mu**2/self.rho_f)*self.R
        else:
            self.M_shaft = (self.L * self.mu**2)/(3.125*self.rho_f*lambertw(0.397*self.R, k = 0)**2)*self.R**2



        return

    

    '''
    Theoretical dissipation curves
    '''

    def enclosed_rotor_stator(self, start, stop):
           '''
           As a first approach to nonlinear fitting, use the results of Daily and Nece (1960) to 
           model the disk face of the cylinder as it approaches the wall. This function assumes the
           shaft of the cylinder remains in the laminar regime, estimating the dissipation in watts/m^2
           
           start: the start point for the theoretical curve for the dissipation in meters [m]
           stop: the stop point for the theoretical curve for the dissipation in meters [m]
           '''   
           
           s_array = np.linspace(start, stop, 100)
           G_1 = s_array/self.b
           G_2 = (self.elle - self.L - s_array)/self.b

           G_crit = 1.62*self.R**(-5/11) 

           n = 3.5
           M_disk_1 = (self.b*self.mu**2)/(2*self.rho_f)*self.R**2*(((np.pi/self.R)*G_1**-1)**n + ((1.85/np.sqrt(self.R))*G_1**0.1)**n)**(1/n)
           M_disk_2 = (self.b*self.mu**2)/(2*self.rho_f)*self.R**2*(((np.pi/self.R)*G_2**-1)**n + ((1.85/np.sqrt(self.R))*G_2**0.1)**n)**(1/n)
           
           dissipation = M_disk_1*self.Omega/(np.pi*self.b**2) + M_disk_2*self.Omega/(np.pi*self.b**2) + self.M_shaft*self.Omega/(2*np.pi*self.b*self.L)
   
           return s_array, dissipation, G_crit


    
    def open_rotor_stator(self, start, stop):
        '''
        As a second approach to nonlinear fitting, modify the results of Von Karman (1921) to 
        model the disk face of the cylinder as it approaches the wall. This function assumes the
        shaft of the cylinder remains in the laminar regime, estimating the dissipation in watts/m^2
        
        start: the start point for the theoretical curve for the dissipation in meters [m]
        stop: the stop point for the theoretical curve for the dissipation in meters [m]
        '''        

        s_array = np.linspace(start, stop, 100)
        G_1 = s_array/self.b
        G_2 = (self.elle - self.L - s_array)/self.b

        #Best fit parameters from ODE solution
        c_0 = 1.659
        c_1 = 0.368
        n = 3.790

        G_crit = 1.595*self.R**-0.366

        M_disk_1 = (self.b*self.mu**2)/(2*self.rho_f)*self.R**2*((np.pi/(G_1*self.R))**n + (c_0*G_1**c_1/np.sqrt(self.R))**n)**(1/n)
        M_disk_2 = (self.b*self.mu**2)/(2*self.rho_f)*self.R**2*((np.pi/(G_2*self.R))**n + (c_0*G_2**c_1/np.sqrt(self.R))**n)**(1/n)
        
        dissipation = M_disk_1*self.Omega/(np.pi*self.b**2) + M_disk_2*self.Omega/(np.pi*self.b**2) + self.M_shaft*self.Omega/(2*np.pi*self.b*self.L)

        return s_array, dissipation, G_crit
   


    '''
    Moment Coefficient Numerical ODE Fitting
    '''
    def disk_moment_coefficient(self, start, stop):
        '''
        Compute the moment coefficient of a rotating disk for a range of gap values
        '''

        #sweep distance
        s_array = np.linspace(start, stop, 100)

        s_sound = []
        G_prime = []
        K_guess = 0.3

        
        for s in s_array:
            #setup mesh
            s_1 = s * np.sqrt(self.rho_f * self.Omega/self.mu)
            z = np.linspace(0, s_1, 100)

            #initial guess
            y_guess = np.zeros((5, z.size))
            y_guess[0] = 0.2 * z * np.exp(-z) # F guess
            y_guess[2] = np.exp(-z)            # G guess
            y_guess[4] = -0.5 * z              # H guess

            sol = solve_bvp(disk_wall_ode, disk_wall_bc, z, y_guess, p = [K_guess])

            if sol.success:
                s_sound.append(s)
                G_prime.append(sol.y[3][0])

        G_prime = np.asarray(G_prime)
        C_m = -np.pi*G_prime/np.sqrt(self.R)


        return np.asarray(s_sound), C_m



    def disk_moment_fit(self, start, stop):
        '''
        Find a fit for the regimes of moment coefficients with the lowest chi squared
        '''

        s_array, C_m = self.disk_moment_coefficient(start, stop)

        popt, _ = curve_fit(lambda x, c_0, c_1, n: ((np.pi*self.b/(x*self.R))**n + (c_0*(x/self.b)**c_1/np.sqrt(self.R))**n)**(1/n), s_array, C_m, p0 = [1, 1, 1])

        best_fit = ((np.pi*self.b/(s_array*self.R))**popt[2] + (popt[0]*(s_array/self.b)**popt[1]/np.sqrt(self.R))**popt[2])**(1/popt[2])

        return s_array, C_m, best_fit, popt


'''
ODE and curve fit Equations, separate from Class
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
