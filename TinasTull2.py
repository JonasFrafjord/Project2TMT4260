# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 19:29:56 2016

@author: tinabe
"""
# -*- coding: utf-8 -*-
"""
Tina Bergh
Jonas Frafjord
Jonas Kristoffer Sunde
TMT4260 Modellering av Fasetransformasjonar
Team 1
Project 2, Part 2B
"""
import sys
import numpy as np
import scipy.sparse
import scipy.special
from matplotlib import pyplot as plt
import math
import time

start_time = time.time()

# Global variables
"Constants"
pi = np.pi
R = 8.314 # [J/(K*mol)] Univ. Gas Constant  
NA = 6.022*10**23 # Avogadro's constant, [particles/mol]
"Isothermal annealing at T = 400 [C]"
T_K = 273.15 # Deg K at 0 deg C
T_i = 400.0+T_K # [K]

"From table 1 in Bjørneklett"
C_star=2.17e1 # wt%/100
DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0 # [at%]
C_0=0.0  # [at]      
C_i = C_star*np.exp(-DeltaH/(R*T_i))

#Diffusivity for T_i
D_i = D_0*np.exp(-Q/(R*T_i))


"Spacial and temporal discretisation"
N = 300 # Number of spacial partitions of bar
L = 1.5 # [um] Length of barH = 30.0 
#t_i = 0.1 # senconds for isothermal annealing
t_i = 10 # senconds for isothermal annealing
#T1 = 1e3+T_K # [K] Temperature           
T_1 = T_i # [K] Temperature           
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix."
# Calculating and plotting concentration profile for the spatial range [-1,+1] mm after 20 s annealing at 400 deg C

# Calculates diffusivity
def Diffusivity(T):
    return D_0*np.exp(-Q/(R*T))

def k_fun(C_it):
    return 2*(C_it-C_0)/(C_p-C_0)
    
#Analytical normalized (relative) radius of spherical precipitate (3D case)    
def Bf(k,t,D,r_init):
#Check if short time or long term solution should be used.     
#    LongTerm = -k*D/(2*(r_0-k*D*t/(2*r_0)-k/(np.sqrt(D*t/pi))))
#    ShortTerm = -k/2*np.sqrt(D/(pi*t))   
  #  if  LongTerm > ShortTerm*10:
   #     return np.sqrt(1-k*D*t/B_init**2) # Long time solution
    #NB! Only short time in exercise, do not need to check. 
    return (r_init-k*D*t/(2*r_init)-k*np.sqrt((D*t)/pi))/r_0 # Short time solution

#Numerical, Normalized Volume Fraction of Spherical Particle for Two-Step annealing, LONG TIMES
def NextVolFracNum():
    return 
    
#Analytical, Normalized Volume Fraction of Spherical Particle for Two-Step annealing, LONG TIMES    
def VolFrac(k_temp,D_temp,t_temp):
    return (1 - k_temp*D_temp*t_temp/r_0**2)**(2/3)  
    
#print(C_i, np.sqrt(D_1),k_fun(C_i), "C_i, D_1, k")
#print(pi/D_1*B_0**2/k_fun(C_i)**2, "Tid")

#Calculate the concentration on the particle surface at the temperature T_i
def C_i_f(Ttemp):
    return C_star*np.exp(-DeltaH/(R*Ttemp))
#print(Csurf(T_i))

# Use for non-isothermal
#def C(x,r,T,D,t):
#    return Csurf(T)-(Csurf(T)-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
def C(z,r,Temp,D,t):
    if((z-dx/2) <= r):
        return C_p
    return C_0-(C_i_f(Temp)-C_0)*(r/z)*(1-scipy.special.erf((z-r)/(2.0*np.sqrt(D*t))))
            
#print(C(3,r_0,T_i,Diffusivity(T_i),t_i))

#print(C(3,r_0,T_i,Diffusivity(T_i),20))


 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    Conc = [C(i,r_0,T_i,Diffusivity(T_i),t_i) for i in x_bar]
    plt.figure()    
    plt.plot(x_bar,Conc) #label='Si'
    plt.xlim(0, L)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('r [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('3D Analytic concentration profile of Si after %d seconds annealing at %d K' % (t_i, T_i))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 18})
    return Conc
       
def Nextr(D_temp, k_temp, t_temp, dt_temp, r_prev):
    r_temp = r_prev - dt_temp*k_temp/(2*r_0)*(D_temp/r_prev + np.sqrt(D_temp/(pi*t_temp)))
    if r_temp < 0:
        return 0
    return r_temp
    
#def t_star(k,D): trenger ikke denne?... 
#    return t_r*(k_r*B_0)**2*D_r/(D*(k*B_0r)**2)
    
# Create diagonal and sub/super diagonal for tridiagonal sparse matrix
def createSparse(DTemp1, D_ZT):
    sup = [alpha*DTemp1/D_ZT*((1/(i+1))-1) for i in range(N)]    # sub and super is equivalent for this finite difference scheme
    sub = [alpha*DTemp1/D_ZT*(1-(1/(i+1))) for i in range(N)] 
    diag = np.zeros(N+1)+1-2*alpha*DTemp1/D_ZT  # diagonal
    return scipy.sparse.diags(np.array([sub,diag,sup]), [-1,0,1])

# Calculation of new concentration profile per time increment
def nextTime(CVecT, AMatT):
    return np.dot(CVecT,AMatT)

# Calculation of new concentration profile per time increment (sparse matrix) 
def nextTimeSparse(CVecT, ASparseT):
    return CVecT*ASparseT
    
#def saveFig(xVecT,CVecT,timeT,tempT,figNameT):
def saveFig(xVecT,CVecT,timeT,figNameT):
    plt.plot(xVecT, CVecT, label='Cu') # NB! Change metal name if calc. anim. for Ni
    plt.xlim(-1, 1)
    plt.ylim(0, 1.1)
    plt.xlabel('x [mm]')
    plt.ylabel('Concentration [mol/mm]')
    plt.title('Concentration profile after %d hours annealing at %d K' % (timeT, T_anneal))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 18})
    plt.savefig(figNameT,transparant=True)

def fin_diff(T1,T2,var):
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
 
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    D_Z = max(D_1,D_2) 
 
    # Spatial discretisation is global
    # Temporal discretisation
 
    dt = alpha*dx**2/D_1
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt) # mesh points in time

    # Create initial concentration vectors
    U = np.append(np.zeros(int(r_0/dx)+1)+1,np.zeros(N-int((r_0)/dx)))
    U[int(r_0/dx)+1] = 0.5
   
    # Create the sparse matrix
    Sparse=createSparse(D_1,D_1)
    
    #Solve for every timestep. RSR=relative sphere radius
    RSR_num = np.zeros(np.size(t))
    RSR_num[0] = 1.0
    RVF_isokin = np.zeros(np.size(t))
    RSR_num = np.zeros(np.size(t))
    RSR_num[0] = 1.0
    D_RSR = D_1
    r_RSR = B_0
    t_RSR = 0
    T_RSR = T1
    ij = 0
    k_RSR = k_fun(C_i_f(T_RSR))
    
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
        U[N] = 0
        RSR_temp = Bf(k_fun(C_i),dt*i,D_RSR,r_0)
        if (RSR_temp < var and ShouldChange):
            D_RSR = D_2
            r_RSR= RSR_temp*r_RSR
            t_RSR = dt*i
            T_RSR = T2
            k_RSR = k_f(C_i_f(T_RSR))
            ShouldChange = False
            sparse = createSparse(D_2,D_Z)
            ij = 0
        
        if (RSR_temp > 0):
            RSR_isokin[i] = RSR_temp
        if (i == 0):
            ij = ij+1
            continue
        RSR_num[i] = Nextr(D_1,k_1, (i+1)*dt,dt,RSR_num[i-1])
        ij = ij+1
    plt.figure()    
    plt.plot(x_bar,U)
    plt.ylim(-1.1,1.1)
    plt.title('Numerical concentration profile in 3D for isothermal annealing at %d K for %d seconds' %(T_i,t_i))
    plt.figure()
    plt.plot(t, sphere_radius)
    plt.title('Analytical normalized sphere radius (3D) for isothermal annealing at %d K for %d seconds' %(T_i,t_i))
    plt.figure()
    plt.plot(t, RSR_num)
    plt.title('Numerical normalized sphere radius (3D) for isothermal annealing at %d K for %d seconds' %(T_i,t_i))

def main(argv):
    analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
    finite_diff() # Calc and plot concentration profiles, finite differences
    plt.show() 
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
