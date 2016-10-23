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
Project 2, Part 2A
3D
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
T_low = 400.0+T_K
T_hi = 430.0+T_K

"From table 1 in Bjørneklett"
C_star=2.17e3 # wt%/100
DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0e2 # [at%]
C_0=0.0  # [at]      

#Diffusivity for T_i
D_i = D_0*np.exp(-Q/(R*T_i))


"Spacial and temporal discretisation"
N = 300 # Number of spacial partitions of bar
L = 1.5 # [um] Length of barH = 30.0 
#t_i = 0.1 # senconds for isothermal annealing
t_i = 1e-1 # senconds for isothermal annealing
#T1 = 1e3+T_K # [K] Temperature           
T_1 = T_i # [K] Temperature           
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix."
# Calculating and plotting concentration profile for the spatial range [-1,+1] mm after 20 s annealing at 400 deg C

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
   
def R_f(k,t,D,B_init):
    return (B_init-k*(np.sqrt((D*t)/pi)))/B_0
#print(C_i, np.sqrt(D_1),k_fun(C_i), "C_i, D_1, k")
#print(pi/D_1*B_0**2/k_fun(C_i)**2, "Tid")

#Calculate the concentration on the particle surface at the temperature T_i
def Csurf(T):
    return C_star*np.exp(-DeltaH/(R*T))
#print(Csurf(T_i))

# Use for non-isothermal
#def C(x,r,T,D,t):
#    return Csurf(T)-(Csurf(T)-C_0)*scipy.special.erf((x-r)/(2.0*np.sqrt(D*t)))
def CAnal(r,R,T,D,t,C_i_T):
    if((r-dx/2) <= R):
        return C_i_T
    return C_0-(C_i_T-C_0)*(R/r)*(1-scipy.special.erf((r-R)/(2.0*np.sqrt(D*t))))
            
#print(C(3,r_0,T_i,Diffusivity(T_i),t_i))

#print(C(3,r_0,T_i,Diffusivity(T_i),20))


 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    C_i_T = C_i_f(T_i)
    Conc = [CAnal(i,r_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
    plt.plot(x_bar,Conc) #label='Si'
    plt.xlim(0, L)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('r [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('3D Analytic concentration profile of Si after %d seconds annealing at %d K' % (t_i, T_i))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 18})
    return Conc
       

    
#def t_star(k,D): trenger ikke denne?... 
#    return t_r*(k_r*B_0)**2*D_r/(D*(k*B_0r)**2)
    
# Create diagonal and sub/super diagonal for tridiagonal sparse matrix
def createSparse(DTemp1, D_ZT):
    sup = [alpha*DTemp1/D_ZT*(1+(1/(i+1))) for i in range(N)]    # sub and super is equivalent for this finite difference scheme
    sub = [alpha*DTemp1/D_ZT*(1-(1/(i+1))) for i in range(N)] 
    diag = np.zeros(N+1)+(1-2*alpha*DTemp1/D_ZT)  # diagonal
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

def NextR(D_temp, k_temp, t_temp, dt_temp, R_prev):
    R_temp = R_prev
    return R_temp

def fin_diff(T1,T2,RSR_ch):
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
    #Diffusivity at 1.st stage and 2.nd stage
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    D_Z = max(D_1,D_2)

    # Temporal discretisation
    dt = alpha*dx**2/D_Z       # D_hi will give the lowest dt, use it to be sure we respect the stability criterion
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
    # Create initial concentration vectors
    index_cutoff = round(r_0/dx)+1
    U = np.append(np.zeros(index_cutoff)+1,np.zeros(N-index_cutoff+1))
    U[index_cutoff] = 0.5

    # Create diag, sub and super diag for tridiag
    Sparse = createSparse(D_1,D_Z)

    #Solve for every timestep
    RSR_isokin = np.zeros(np.size(t))
    RSR_num = np.zeros(np.size(t))
    RSR_num[0] = 1.0
    D_RSR = D_1
    B_RSR = B_0
    t_RSR = 0
    T_RSR = T1
    i_time = 0
    C_i_RSR = C_i_f(T_RSR)
    k_RSR = k_f(C_i_RSR)
    print('k_RSR is {}'.format(k_RSR))
    print('D1 is {0}, and D2 is {1}'.format(D_1,D_2))
    print('b0 is {}'.format(B_RSR))

    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        for j in range(round(r_0/dx)+1):
            U[j] = C_p # inf BC
            U[index_cutoff] = C_i_RSR
        U[N] = 0
        RSR_temp = R_f(k_f(C_i_RSR),dt*i_time,D_RSR,B_RSR)
        if (RSR_temp < RSR_ch and ShouldChange):
            print("Yes")
            D_RSR = D_2
            B_RSR = RSR_temp
            i_time = 0
            ShouldChange = False
            sparse = createSparse(D_2,D_z)
        if (RSR_temp > 0):
            RSR_isokin[i] = RSR_temp
        i_time = i_time +1
    print(i_time,Nt)
    plt.figure()
    plt.plot(x_bar,U)
 #   plt.figure()
 #   plt.plot(t,RSR_isokin)
 #   plt.ylim(0,1.1)


def main(argv):
    analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
 #   finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
    fin_diff_two_step(T_low,T_low,0.3)
 #   NextBnum()
    plt.show()
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
    
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
