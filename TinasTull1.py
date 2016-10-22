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
#NB!
C_star=2.17e3 # [wt.%]
C_star2=2.17e1 # [wt.%]/100

DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0e2 # [wt.%] <--100.0
C_0=0.0  # [wt.%]
#NB!
C_p2=1.0 # [wt.%]
C_02=0.0  # [wt.%]       

#Diffusivity for T_i
D_i = D_0*np.exp(-Q/(R*T_i))

"Spatial and temporal discretisation"
N = 300 # Number of spatial partitions of bar
L = 1.5 # [um] Length of bar 
#t_i = 0.1 # seconds for isothermal annealing
t_i = 15 # seconds for isothermal annealing
#T1 = 1e3+T_K # [K] Temperature           
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix."
# Calculating and plotting concentration profile for the spatial range [0,1.5] um after 20 s annealing at 400 deg C

C = C_star*np.exp(-DeltaH/(R*T_i))
print('Interface concentration: %e wt. percentage\n' % C)

# Concentration at interface at temperature Ttemp
def C_i_f(Ttemp):
    C = C_star*np.exp(-DeltaH/(R*Ttemp))
    #print('Concentration at interface at temperature %.1f K: %e mol/um' % (Ttemp,C))
    return C

# Calculates diffusivity
def Diffusivity(T):
    return D_0*np.exp(-Q/(R*T))

def k_f(C_it):
    #return 2*(C_it-C_0)/(C_p-C_0)
    return 2*(C_it-C_0)/(C_p-C_it) # NB! Whelan definition
    
#def Bf(k,t,D,B_init,t_init):
#   return (B_init-k*(np.sqrt(D*t/pi)-np.sqrt(D*t_init/pi)))/B_0

def Bf(k,t,D,B_init):
    return (B_init-k*(np.sqrt(D*t/pi)))/B_0
    
T_dis = pi/D_i*B_0**2/k_f(C_i_f(T_i))**2
print('Time to completely dissolve the particle at T_i = {0:.0f} K: {1:.3f} s\n' .format(T_i,T_dis))

def CAnal(x,r,T,D,t,C_i_T):
    if((x-dx/2) <= r):
        return C_i_T
    #return C_p-(C_p-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
    # Use for non-isothermal
    return C_i_T-(C_i_T-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))

 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    C_i_T = C_i_f(T_i)
    Conc_1 = [CAnal(i,r_0,T_i,D_i,t_i/4,C_i_T) for i in x_bar]
    Conc_2 = [CAnal(i,r_0,T_i,D_i,t_i/2,C_i_T) for i in x_bar]
    Conc_3 = [CAnal(i,r_0,T_i,D_i,t_i*3/4,C_i_T) for i in x_bar]
    Conc_4 = [CAnal(i,r_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
    plt.plot(x_bar,Conc_1,',', label='After {:.2f}s'.format(t_i/4)) #label='Si'
    plt.plot(x_bar,Conc_2,',', label='After {:.2f}s'.format(t_i/2)) #label='Si'
    plt.plot(x_bar,Conc_3,',', label='After {:.2f}s'.format(t_i*3/4)) #label='Si'
    plt.plot(x_bar,Conc_4,',', label='After {:.2f}s'.format(t_i)) #label='Si'
    plt.xlim(0, L)
    plt.ylim(0, 1.1)
    plt.xlabel('x [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('Analytic concentration profile of Si  after %d seconds annealing at %d K' % (t_i, T_i))
 #   plt.legend(bbox_to_anchor=(0.2,1))
    plt.legend()
    plt.rcParams.update({'font.size': 18})
    return Conc_4
       
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

def fin_diff(T1,T2,RPT_ch):
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
    
    # Spatial discretization
    # Temporal discretisation
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    D_Z = max(D_1,D_2)
    dt = alpha*dx**2/D_Z        # D_Z will give the lowest dt, use it to be sure we respect the stability criterion
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
 # Create initial concentration vectors
    index_cutoff = round(r_0/dx)+1
    U = np.append(np.zeros(index_cutoff)+1,np.zeros(N-index_cutoff+1))
    U[index_cutoff] = 0.5 # Since initial value is undefined at x = 0, we set it to 0.5 which also smoothens the graph
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
    i_time = 0
    C_i_RPT = C_i_f(T_RPT)
    k_RPT = k_f(C_i_RPT)
    print('k_RPT is {}'.format(k_RPT))
    print('D1 is {0}, and D2 is {1}'.format(D_1,D_2))
    print('b0 is {}'.format(B_RPT))
 #   plt.figure()
    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(index_cutoff):
            U[j] = C_i_RPT # inf BC
        U[N] = 0
        RPT_temp = Bf(k_RPT,dt*i_time,D_RPT,B_RPT)
        #RPT_temp = Bf(k_RPT,dt*i_time,D_RPT,B_RPT,t_RPT)
        if (RPT_temp < RPT_ch and ShouldChange):
            D_RPT = D_2
            B_RPT = RPT_temp*B_RPT
            t_RPT = dt*i
            T_RPT = T2
            k_RPT = k_f(C_i_f(T_RPT))
            ShouldChanged = False
            sparse = createSparse(D_2,D_Z)
            i_time = 0
        if (RPT_temp > 0):
            RPT_isokin[i] = RPT_temp
        if (i == 0):
            i_time = i_time+1
            continue
        RPT_num[i] = NextB(D_RPT,k_RPT, (i_time+1)*dt,dt,RPT_num[i-1])
        i_time = i_time+1
        if any(i*dt<t_i*itt+dt/2 and i*dt>t_i*itt-dt/2 for itt in [1/4,1/2,3/4]):
            plt.plot(x_bar, U, label='After {:.2f}s'.format(i*dt))
 #   exit()
    plt.plot(x_bar,U,label='After {:.2f}s'.format(t_i))
    plt.ylim(0, 1.1)
    plt.legend()
    plt.figure()
    plt.plot(t,RPT_isokin)
    plt.ylim(0,1.1)
    #plt.figure()
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
    analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
    #finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
    fin_diff(T_low,T_low,0)
#    fin_diff(T_hi,T_low,0.3)
    #fin_diff(T_low,T_hi,0.3)
#    fin_diff(T_hi,T_low,0.7)
#    fin_diff(T_low,T_hi,0.7)
    plt.show()
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
    
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
