import numpy as np
import scipy
from scipy.linalg import solve_banded
from scipy import integrate
import ClimateUtils as clim
import matplotlib.pyplot as plt

def P2( x ):
    '''The second Legendre polynomial.'''
    return 1. / 2. * (3. * x**2 - 1. )

class Radeq:
    # convention : index 0 = TOA, index n-1 = surface air, index n = surface
    def __init__(self, num_lev = 41):
        self.n = num_lev
        self.sigma = np.linspace(0,1,self.n)
        self.p = self.sigma*100000
        self.T = np.ones(self.n)*200
        self.dp = 100000/self.n
        self.dF = np.ones(self.n-1)
        
    def compute_tau_LW(self,tau_surf = 3):
        self.tau_LW = np.ones_like(self.sigma)
        for i in range(0,self.n):
            self.tau_LW[i]=tau_surf*(0.2*self.sigma[i]+0.8*np.power(self.sigma[i],4))
        
        # Transmissivity of layer
        self.dtau_LW = np.ones(self.n-1)
        for i in range(0,self.n-1):
            self.dtau_LW[i]=np.exp(-(self.tau_LW[i+1]-self.tau_LW[i]))
    
    def compute_tau_SW(self, tau_surf = 0.22):
        self.tau_SW = np.ones_like(self.sigma)
        for i in range(0,len(self.sigma)):
            self.tau_SW[i]=tau_surf*np.power(self.sigma[i],2)
            
    def compute_dw_LW(self):
        self.dw_LW = np.zeros(self.n)
        self.dw_LW[0]=0
        for i in range(0,self.n-1):
            self.dw_LW[i+1]=self.dw_LW[i]*self.dtau_LW[i]+clim.sigma*np.power(self.T[i],4)*(1- self.dtau_LW[i])
            
    def compute_dw_SW(self,lat=0):
        insolation = 0.25*1360*(1+1.4*0.25*(1-3*np.sin(lat/180*np.pi)**2))
        self.dw_SW = np.zeros(self.n)
        for i in range(0,len(self.dw_SW)):
            self.dw_SW[i]=insolation * np.exp(-self.tau_SW[i])
            
    def compute_uw_LW(self):
        self.uw_LW = np.zeros(self.n)
        self.uw_LW[self.n-1]=clim.sigma*np.power(self.T[self.n-1],4)
        for i in range(self.n-2,-1,-1):
            self.uw_LW[i]=self.uw_LW[i+1]*self.dtau_LW[i]+clim.sigma*np.power(self.T[i],4)*(1-self.dtau_LW[i])
            
    def compute_uw_SW(self, albedo=0.3):
        self.uw_SW = np.zeros(self.n)
        for i in range(0,len(self.uw_SW)):
            self.uw_SW[i]=albedo * self.dw_SW[i]
    
    def compute_netflux(self):
        self.net_LW = self.uw_LW - self.dw_LW
        self.net_SW = self.uw_SW - self.dw_SW
        self.netflux = self.net_LW + self.net_SW
        
    def update_T(self, num_days=1):
        for i in range(0,self.n-1):
            self.dF[i] = self.netflux[i+1]-self.netflux[i]
        for i in range(0,self.n-1):
            self.T[i] = self.T[i] + (clim.g/clim.cp)*num_days*clim.seconds_per_day*(self.dF[i]/self.dp)
        # increment surface temperature
        self.T[self.n-1]=self.T[self.n-1] + (num_days/(clim.cw*clim.rho_w))*clim.seconds_per_day*(-self.netflux[self.n-1])
        
    def run(self,tau0_LW=3,tau0_SW=0.22,lat=0,albedo=0.3,num_days=0.1,n_iter=100):
        self.compute_tau_LW(tau0_LW)
        self.compute_tau_SW(tau0_SW)
        for i in range(n_iter):
            self.compute_dw_LW()
            self.compute_dw_SW(lat=lat)
            self.compute_uw_LW()
            self.compute_uw_SW(albedo=albedo)
            self.compute_netflux()
            self.update_T(num_days=num_days)
