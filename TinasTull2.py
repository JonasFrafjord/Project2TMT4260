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
C_star=2.17e3 # wt%
DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0e2 # [at%]
C_0=0.0  # [at]      

"Spacial and temporal discretisation"
N = 300 # Number of spacial partitions of bar
L = 1.5 # [um] Length of barH = 30.0 
t_i = 1e1 # senconds for isothermal annealing
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
    return 2*(C_it-C_0)/(C_p-C_it) # NB! Whelan definition
   
#Analytical normalized (relative) radius of spherical precipitate (3D case)    
def R_f(k,t,D,r_init):
#Check if short time or long term solution should be used.     
#    LongTerm = -k*D/(2*(r_0-k*D*t/(2*r_0)-k/(np.sqrt(D*t/pi))))
#    ShortTerm = -k/2*np.sqrt(D/(pi*t))   
  #  if  LongTerm > ShortTerm*10:
   #     return np.sqrt(1-k*D*t/B_init**2) # Long time solution
    #NB! Only short time in exercise, do not need to check. 
    return (r_init-k*D*t/(2*r_init)-k*np.sqrt((D*t)/pi))/r_0 # Short time solution

def NextR(k_temp, t_temp, dt_temp, D_temp, r_init, r_prev):
    if k_temp*D_temp*t_temp > r_0**2: return 0
    if t_temp < 0:
        r_temp = r_prev - (dt_temp*k_temp/2*(D_temp/r_prev + np.sqrt(D_temp/(pi*t_temp))))
    else:
        r_temp = np.sqrt(r_0**2-k_temp*D_temp*t_temp)
    return r_temp

#Isokinetical solution, Normalized Volume Fraction of Spherical Particle for Two-Step annealing, LONG TIMES    
def VolFrac(k_temp,t_temp,D_temp,r_init):
    if k_temp*D_temp*t_temp/r_init**2>1: return 0
    return (1.0 - k_temp*D_temp*t_temp/r_init**2)**(3.0/2.0)  

def CAnal(r,R,T,D,t,C_i_T):
    if((r-dx/2) <= R):
        return C_i_T
    return C_0+(C_i_T-C_0)*(R/r)*(1-scipy.special.erf((r-R)/(2.0*np.sqrt(D*t))))

            
#print(C(3,r_0,T_i,Diffusivity(T_i),t_i))

#print(C(3,r_0,T_i,Diffusivity(T_i),20))


 # Plot the analytical solution with constant diffusivity (D(x) = D = const.)
def AnalConc():
    C_i_T = C_i_f(T_i)
    D_i = Diffusivity(T_i)
    Conc = [CAnal(i,r_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
    plt.plot(x_bar,Conc) #label='Si'
    plt.xlim(0, L)
    plt.ylim(-1.1, 1.1)
    plt.xlabel('r [um]')
    plt.ylabel('Concentration [mol/um]')
    plt.title('3D Analytic concentration profile of Si after %d seconds annealing at %d K' % (t_i, T_i))
    #plt.rcParams.update({'font.size': 18})
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
    

def fin_diff(T1,T2,RSR_ch):
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
   #Diffusivity at 1.st stage and 2.nd stage
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    D_Z = max(D_1,D_2)

    #Variables needed in this module
    D_RSR = D_1
    R_RSR = r_0
    t_RSR = 0
    T_RSR = T1
    i_time = 0
    C_i_RSR = C_i_f(T_RSR)
    k_RSR = k_f(C_i_RSR)


    # Temporal discretisation
    dt = alpha*dx**2/D_Z       # D_hi will give the lowest dt, use it to be sure we respect the stability criterion
    Nt = math.ceil(t_i/dt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
    # Create initial concentration vectors
<<<<<<< HEAD
    index_cutoff = round(r_0/dx)+1
=======
    index_cutoff = round(r_0/dx)
>>>>>>> 1ad88209eef06324c40be65ca12a9a93f9136efb
    U = np.append(np.zeros(index_cutoff)+C_p,np.zeros(N-index_cutoff+1)+C_0)
    U[index_cutoff] = C_i_RSR

    # Create diag, sub and super diag for tridiag
    Sparse = createSparse(D_1,D_Z)

    #Solve for every timestep
    
    RSR_num = np.zeros(np.size(t))
    RSR_num[0] = r_0
    RSR_anal = np.zeros(np.size(t))
    VF_num = np.zeros(np.size(t))
    VF_num[0] = 1.0
    VF_isokin = np.zeros(np.size(t))      
    
    print('k_RSR is {}'.format(k_RSR))
    print('D1 is {0}, and D2 is {1}'.format(D_1,D_2))
    print('b0 is {}'.format(R_RSR))

    for i in range(Nt):
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions       
        U[N] = C_0
        RSR_anal[i] = R_f(k_RSR,dt*i_time,D_RSR,R_RSR)
        RSR_num_temp = NextR(k_RSR,dt*i_time,dt,D_RSR,R_RSR,RSR_num[i-1])
        VF_iso_temp = VolFrac(k_RSR,dt*i_time,D_RSR,R_RSR)
        
        if (RSR_num_temp < RSR_ch and ShouldChange):
            print('T1 and T2 are different')
            D_RSR = D_2
            R_RSR = RSR_num_temp
            i_time = 0
            ShouldChange = False
            sparse = createSparse(D_2,D_Z)
 #       if (VF_iso_temp > 0):
            VF_isokin[i] = VF_iso_temp
 #       if (RSR_num_temp > 0):
            VF_num[i]= (RSR_num_temp/r_0)**3
            RSR_num[i] = RSR_num_temp
        i_time = i_time +1
    plt.figure()
<<<<<<< HEAD
    plt.plot(x_bar[index_cutoff::],U[index_cutoff::])
    plt.ylim(-1.1,1.1)
    plt.figure()
    plt.plot(t,RSR_num)
    plt.plot(t,RSR_anal,',')
    plt.ylim(0,1.1)
    plt.figure()
    plt.plot(t,VF_num)
    plt.plot(t,VF_isokin,',')
    plt.ylim(0,1.1)
    plt.xlim(0,21)
=======
    plt.plot(x_bar,U)
    #plt.ylim(-1.1,1.1)
    plt.figure()
    plt.plot(t,RSR_num)
#    plt.plot(t,RSR_anal,',')
#    plt.ylim(0,1.1)
#    plt.figure()
#    plt.plot(t,VF_num)
#    plt.plot(t,VF_isokin,',')
#    plt.ylim(0,1.1)
#    plt.xlim(0,21)
>>>>>>> 1ad88209eef06324c40be65ca12a9a93f9136efb

def main(argv):
    analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
 #   finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
    fin_diff(T_low,T_low,0.3)
#    fin_diff(T_low,T_hi,0.3)
 #   NextBnum()
    plt.show()
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
