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
T_low = 400.0+T_K # [K]
T_hi = 430.0+T_K # [K]

"From table 1 in Bjørneklett"
C_star=2.17e1 # wt.%/100
DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0 # [wt.%]
C_0=0.0  # [wt.%]      

#Diffusivity for T_i
D_i = D_0*np.exp(-Q/(R*T_i))

<<<<<<< HEAD
"Spatial and temporal discretisation"
N = 300 # Number of spatial partitions of bar
L = 1.5 # [um] Length of bar 
#t_i = 0.1 # seconds for isothermal annealing
t_i = 15 # seconds for isothermal annealing
=======


"Spacial and temporal discretisation"
N = 100 # Number of spacial partitions of bar
L = 1.5 # [um] Length of barH = 30.0 
#t_i = 0.1 # senconds for isothermal annealing
t_i = 20 # senconds for isothermal annealing
>>>>>>> 4c36bdb643cfd1ea904a16d0c0c34593086b72d2
#T1 = 1e3+T_K # [K] Temperature           
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix."
# Calculating and plotting concentration profile for the spatial range [0,1.5] um after 20 s annealing at 400 deg C

# Concentration at interface at temperature Ttemp
def C_i_f(Ttemp):
    C = C_star*np.exp(-DeltaH/(R*Ttemp))
    #print('Concentration at interface at temperature %.1f K: %e mol/um' % (Ttemp,C))
    return C

# Calculates diffusivity at temp. T
def Diffusivity(T):
    return D_0*np.exp(-Q/(R*T))

# Calculates concentration coefficient k_f
def k_f(C_it):
    return 2*(C_it-C_0)/(C_p-C_0)
    #return 2*(C_0-C_it)/(C_p-C_it)
    
#def Bf(k,t,D,B_init,t_init):
#   return (B_init-k*(np.sqrt(D*t/pi)-np.sqrt(D*t_init/pi)))/B_0

def Bf(k,t,D,B_init):
    return (B_init-k*(np.sqrt(D*t/pi)))/B_0
    
print('Total time req. for complete particle dissol.: %f s' % (pi/D_i*B_0**2/k_f(C_i_f(T_i))**2))

#Calculate the concentration on the particle surface at the temperature T_i
def Csurf(T):
    return C_star*np.exp(-DeltaH/(R*T))
#print(Csurf(T_i))

# Use for non-isothermal
#def C(x,r,T,D,t):
#    return Csurf(T)-(Csurf(T)-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
def CAnal(x,r,T,D,t):
    if((x-dx/2) <= r):
        return C_p
    return C_p-(C_p-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
            

 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    Conc = [CAnal(i,r_0,T_i,D_i,t_i) for i in x_bar]
    plt.plot(x_bar,Conc,',') #label='Si'
    plt.xlim(0, L+.1)
    plt.ylim(0, 1.1)
    plt.xlabel('x [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('Analytic concentration profile of Si  after %d seconds annealing at %d K' % (t_i, T_i))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 16})
    return Conc
       

    
#def t_star(k,D): trenger ikke denne?... 
#    return t_r*(k_r*B_0)**2*D_r/(D*(k*B_0r)**2)
    
# Create diagonal and sub/super diagonal for tridiagonal sparse matrix
def createSparse(DTemp, D_ZT):
    subsup = np.zeros(N)+alpha*DTemp/D_ZT      # sub and super is equivalent for this finite difference scheme
    diag = np.zeros(N+1)+1-2*alpha*DTemp/D_ZT  # diagonal
    return scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])

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

def NextB(D_temp, k_temp, t_temp, dt_temp, B_prev):
    B_temp = B_prev-dt_temp*k_temp*np.sqrt(D_temp/(pi*t_temp))/(B_0*2)
    if B_temp < 0:
        return 0
    return B_temp

<<<<<<< HEAD
def finite_diff():
    # Spatial discretisation is global
    # Temporal discretisation
    T_1 = T_i
    D_1 = D_i
    D_Z = D_1
    k_1 = k_f(C_i_f(T_1))
    dt = alpha*dx**2/D_1
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt) # mesh points in time
    # Create initial concentration vectors
    U = np.append(np.zeros(int(r_0/dx)+1)+1,np.zeros(N-int((r_0)/dx)))
    U[int(r_0/dx)+1] = 0.5
    
    # Create sparse
    Sparse = createSparse(D_1,D_Z)

    #Solve for every timestep
    plate_thickness_bar = np.zeros(np.size(t))
    RPT_num = np.zeros(np.size(t))
    RPT_num[0] = 1.0
    
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
        U[N] = 0
        relative_plate_thickness = Bf(k_f(C_i_f(T_i)),dt*i,D_1,B_0)
        if (relative_plate_thickness > 0):
            plate_thickness_bar[i] = relative_plate_thickness
        if i==0:
            continue
        RPT_num[i] = NextB(D_1,k_1, (i+1)*dt,dt,RPT_num[i-1])
    plt.plot(x_bar,U)
    plt.ylim(0,1.1)
    plt.figure()
    plt.plot(t, plate_thickness_bar)
    plt.figure()
    plt.plot(t, RPT_num)
            

def fin_diff_two_step(T1,T2):
=======

   

def fin_diff(T1,T2,var):
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
>>>>>>> 4c36bdb643cfd1ea904a16d0c0c34593086b72d2
    
    # Spatial discretization
    # Temporal discretisation
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    D_Z = max(D_1,D_2)
    dt = alpha*dx**2/D_Z        # D_hi will give the lowest dt, use it to be sure we respect the stability criterion
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
 # Create initial concentration vectors
    U = np.append(np.zeros(int(r_0/dx)+1)+1,np.zeros(N-int((r_0)/dx)))
    U[int(r_0/dx)+1] = 0.5 # Since initial value is undefined at x = 0, we set it to 0.5 which also smoothens the graph

    # Create sparse
    Sparse = createSparse(D_1,D_Z)

    #Solve for every timestep
    RPT_isokin = np.zeros(np.size(t))
    RPT_num = np.zeros(np.size(t))
    RPT_num[0] = 1.0
    D_RPT = D_1
    B_RPT = B_0
    t_RPT = 0
    T_RPT = T1
    ij = 0
    k_RPT = k_f(C_i_f(T_RPT))
    print(D_Z)
    print(k_RPT)
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
        U[N] = 0
        RPT_temp = Bf(k_RPT,dt*ij,D_RPT,B_RPT)
        #RPT_temp = Bf(k_RPT,dt*ij,D_RPT,B_RPT,t_RPT)
        if (RPT_temp < var and ShouldChange):
            D_RPT = D_2
            B_RPT = RPT_temp*B_RPT
            t_RPT = dt*i
            T_RPT = T2
            k_RPT = k_f(C_i_f(T_RPT))
            ShouldChanged = False
            sparse = createSparse(D_2,D_Z)
            ij = 0
        if (RPT_temp > 0):
            RPT_isokin[i] = RPT_temp
        if (i == 0):
            ij = ij+1
            continue
        RPT_num[i] = NextB(D_RPT,k_RPT, (ij+1)*dt,dt,RPT_num[i-1])
        ij = ij+1
    plt.figure()
    plt.plot(t,RPT_isokin)
    plt.ylim(0,1.1)
    plt.figure()
    plt.plot(t,RPT_num)

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
<<<<<<< HEAD
    analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
    finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
    #fin_diff_two_step(T_hi,T_low)
    #plt.show()
=======
 #   analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
    #Plate_thickness()    
 #   fin_diff(T_low,T_low,0)
#    fin_diff(T_hi,T_low,0.3)
    fin_diff(T_low,T_hi,0.3)
#    fin_diff(T_hi,T_low,0.7)
#    fin_diff(T_low,T_hi,0.7)
    plt.show()
>>>>>>> 4c36bdb643cfd1ea904a16d0c0c34593086b72d2
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
    
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
