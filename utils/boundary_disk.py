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
        x1 = s_array
        x2 = (self.elle - self.L - s_array)

        #Best fit parameters from ODE solution
        c0 = 0.654
        c1 = 0.528
        c2 = 1.895
        c3 = 0.918

        M_disk_1 = self.moment_function(x1, [c0, c1, c2, c3])*(self.b*self.mu**2)/(2*self.rho_f)*self.R**2
        M_disk_2 = self.moment_function(x2, [c0, c1, c2, c3])*(self.b*self.mu**2)/(2*self.rho_f)*self.R**2

        dissipation = M_disk_1*self.Omega/(np.pi*self.b**2) + M_disk_2*self.Omega/(np.pi*self.b**2) + self.M_shaft*self.Omega/(2*np.pi*self.b*self.L)

        return s_array, dissipation
   


    '''
    Moment Coefficient Numerical ODE Fitting
    '''
    def disk_moment_coeff(self, start, stop):
        '''
        Compute the moment coefficient of a rotating disk for a range of gap values
        '''

        #sweep distance
        s_array = np.linspace(start, stop, 100)

        s_sound = []
        G_prime = []
        
        for s in s_array:
            #setup mesh
            eta = np.linspace(0, 1, 100)
            R_s = self.rho_f*self.Omega*s**2/self.mu

            #initial guess
            y_guess = np.zeros((6, eta.size))

            #ode
            sol = solve_bvp(lambda z, y: disk_wall_ode(z, y, R_s), disk_wall_bc, eta, y_guess)

            if sol.success:
                s_sound.append(s)
                G_prime.append(sol.y[5][0])

        G_prime = np.asarray(G_prime)
        s_sound = np.asarray(s_sound)
        C_m = -np.pi*self.b/(s_sound*self.R)*G_prime


        return s_sound, C_m



    def disk_moment_fit(self, start, stop):
        '''
        Find a fit for the regimes of moment coefficients 
        '''

        s_array, C_m = self.disk_moment_coeff(start, stop)

        popt, _ = curve_fit(lambda x, c0, c1, c2, c3: self.moment_function(x, [c0, c1, c2, c3]), s_array, C_m, p0 = [1, 1, 1, 1])
        best_fit = self.moment_function(s_array, popt)

    
        return s_array, C_m, best_fit, popt



    def moment_function(self, x, params):
        '''
        Fitting function used in tandem with disk_moment_fit
        '''
        c0, c1, c2, c3 = params

        G = x/self.b
        zeta = G*np.sqrt(self.R)

        C_couette = np.pi/(G*self.R)*np.exp(-zeta*c0)
        C_free = 1.935/np.sqrt(self.R)*(1 - c1*zeta**c2*np.exp(-zeta*c3))
    
        return C_couette + C_free



'''
ODE and curve fit Equations, separate from Class
'''
def disk_wall_ode(z, y, R_s):
    H, H_prime, H_2prime, H_3prime, G, G_prime = y
    
    dH = H_prime
    d2H = H_2prime
    d3H = H_3prime
    d4H = R_s*(H*H_3prime + 4*G*G_prime)

    dG = G_prime
    d2G = R_s*(H*G_prime - H_prime*G)
    
    return np.vstack((dH, d2H, d3H, d4H, dG, d2G))


def disk_wall_bc(ya, yb):
    '''
    ya corresponds to nu = 0
    yb corresponds to nu = 1

    the scipy BVP function is setup to return residuals
    '''

    return np.array([
        ya[0] - 0.0,  # H(0) = 0   u_z, disk face
        ya[1] - 0.0,  # H'(0) = 0  u_r, disk face
        ya[4] - 1.0,  # G(0) = 1   u_ϕ, disk face
        yb[0] - 0.0,  # H(1) = 0   u_z, wall
        yb[2] - 0.0,  # H'(1) = 0  u_r, wall
        yb[4] - 0.0,  # G(1) = 0   u_ϕ, wall
    ])




