# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:29:39 2016

@author: tinabe
"""

# -*- coding: utf-8 -*-
"""
Tina Bergh
Jonas Frafjord
Jonas Kristoffer Sunde
TMT4260 Modellering av Fasetransformasjonar
Team 1
Project 2, Part 2A
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
D_1 = D_0*np.exp(-Q/(R*T_i))


"Spacial and temporal discretisation"
N = 300 # Number of spacial partitions of bar
L = 1.5 # [um] Length of barH = 30.0 
#t_i = 0.1 # senconds for isothermal annealing
t_i = 14 # senconds for isothermal annealing
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
    
def Bf(k,t,D,B_init):
    return (B_init-k*(np.sqrt((D*t)/pi)))/B_0
print(C_i, np.sqrt(D_1),k_fun(C_i), "C_i, D_1, k")
print(pi/D_1*B_0**2/k_fun(C_i)**2, "Tid")

#Calculate the concentration on the particle surface at the temperature T_i
def Csurf(T):
    return C_star*np.exp(-DeltaH/(R*T))
#print(Csurf(T_i))

# Use for non-isothermal
#def C(x,r,T,D,t):
#    return Csurf(T)-(Csurf(T)-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
def C(x,r,T,D,t):
    if((x-dx/2) <= r):
        return C_p
    return C_p-(C_p-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
            
print(C(3,r_0,T_i,Diffusivity(T_i),t_i))

print(C(3,r_0,T_i,Diffusivity(T_i),20))


 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    Conc = [C(i,r_0,T_i,Diffusivity(T_i),t_i) for i in x_bar]
    plt.plot(x_bar,Conc,',') #label='Si'
    plt.xlim(0, L)
    plt.ylim(0, 1.1)
    plt.xlabel('x [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('Analytic concentration profile of Si  after %d seconds annealing at %d K' % (t_i, T_i))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 18})
    return Conc
       

    
#def t_star(k,D): trenger ikke denne?... 
#    return t_r*(k_r*B_0)**2*D_r/(D*(k*B_0r)**2)
    
# Create diagonal and sub/super diagonal for tridiagonal sparse matrix
def createSparse(DTemp1, DTemp2):
    subsup = np.zeros(N)+alpha*DTemp/D_1      # sub and super is equivalent for this finite difference scheme
    diag = np.zeros(N+1)+1-2*alpha*DTemp/D_1  # diagonal
    return scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])

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


   

def finite_diff():
    # Spatial discretisation is global
    # Temporal discretisation
    dt = alpha*dx**2/D_1
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt) # mesh points in time

    # Create initial concentration vectors
    U = np.append(np.zeros(int(r_0/dx)+1)+1,np.zeros(N-int((r_0)/dx)))
    U[int(r_0/dx)+1] = 0.5

    
    # Create diag, sub and super diag for tridiag
    subsup = np.zeros(N)+alpha      #sub and super is equivalent
    diag = np.zeros(N+1)+1-2*alpha    #diagonal
    Sparse = scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])

    #Solve for every timestep
    plate_thickness_bar = np.zeros(np.size(t))
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
        U[N] = 0
        relative_plate_thickness = Bf(k_fun(C_i),dt*i,D_1,B_0)
        if (relative_plate_thickness > 0):
            plate_thickness_bar[i] = relative_plate_thickness
    plt.plot(x_bar,U)
    plt.ylim(0,1.1)
    plt.figure()
    plt.plot(t, plate_thickness_bar)
            
def NextBnum():
    dt = alpha*dx**2/D_1
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1)
    Bnum = np.zeros(Nt+1)
    Bnum[0] = 1
    for i in range(Nt):
        Bnum[i+1] = Bnum[i]-dt*k_fun(C_i)*np.sqrt(D_1/(pi*dt*(i+1)))/(B_0*2)
    plt.figure()
    plt.ylim(-0.1,1.1) 
    plt.plot(t,Bnum)

def fin_diff_two_step(T1,T2):
    T1 = 400+T_K
    T2 = 430+T_K
    
    # Spatial discretization
    # Temporal discretisation
    D_low = Diffusivity(T1)
    D_hi = Diffusivity(T2)
    print(D_low/D_hi)
    dt = alpha*dx**2/D_hi        # D_hi will give the lowest dt, use it to be sure we respect the stability criterion
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
 # Create initial concentration vectors
    U = np.append(np.zeros(int(r_0/dx)+1)+1,np.zeros(N-int((r_0)/dx)))
    U[int(r_0/dx)+1] = 0.5 # Since initial value is undefined at x = 0, we set it to 0.5 which also smoothens the graph

    # Create diag, sub and super diag for tridiag
    subsup = np.zeros(N)+alpha*D_low/D_hi      #sub and super is equivalent
    diag = np.zeros(N+1)+1-2*alpha*D_low/D_hi    #diagonal
    Sparse = scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])

    #Solve for every timestep
    plate_thickness_bar = np.zeros(np.size(t))
    NotChanged = True
    D_RPT = D_low
    B_RPT = B_0
    i_time = 0
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
        U[N] = 0
        relative_plate_thickness = Bf(k_fun(C_i),dt*i_time,D_RPT,B_RPT)
        if (relative_plate_thickness < 0.3 and NotChanged):
            print("Yes")
            D_RPT = D_hi
            B_RPT = relative_plate_thickness
            i_time = 0
            NotChanged = False
            subsup = np.zeros(N)+alpha
            diag = np.zeros(N+1)+1-2*alpha
            sparse = scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])
        if (relative_plate_thickness > 0):
            plate_thickness_bar[i] = relative_plate_thickness
        i_time = i_time +1
    print(i_time,Nt)
    plt.figure()
    plt.plot(t,plate_thickness_bar)
    plt.ylim(0,1.1)

def stabilityCheck(exact,approx):
    residual = np.zeros(len(exact))
    residual = np.log10(np.abs(exact-approx))
    #residual = [i/j for i,j in zip(residual,exact)]
    figureX = plt.figure(figsize=(14,10),dpi=600)
    plt.rcParams.update({'font.size': 18})
    plt.plot(np.linspace(-1, 1, N+1),residual,'r')
    plt.xlim(-1, 1)
    plt.xlabel('x [mm]')
    plt.ylabel('Log10(residuals)')
    plt.title('Comparison between analytical and finite difference solutions for C_Cu, N = %d, alpha = %.2f' %  (N,alpha), y=1.03)
    plt.rcParams.update({'font.size': 16})

def main(argv):
 #   analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
 #   finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
 #   fin_diff_two_step()
 #   NextBnum()
    plt.show()
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
    
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
