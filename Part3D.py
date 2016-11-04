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
T_low = 400.0+T_K # [K]
T_hi = 430.0+T_K # [K]

"From table 1 in Bjørneklett"
C_star=2.17e3 # [wt.%]
DeltaH=50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1]
Q = 123.8e3 # [J/mol]
#B_0=1e-3 # [um]
r_0=0.025 # [um]
C_p=1.0e2 # [wt.%]
C_0=0.0  # [wt.%]      

"Spatial and temporal discretisation"
N = 600 # Number of spacial partitions of bar
L = 1.5 # [um] Length of bar 
t_i = 10 # seconds for isothermal annealing
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


#How often should the Sparse matrix be updated
upSparse = 200

"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix."
# Calculating and plotting concentration profile for the spatial range [0,L] um after t_i s annealing at T_i deg C

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
   
# Analytical normalized (relative) radius of spherical precipitate (3D case)   
# Check if short time or long term solution should be used. 
def R_f(k,t,D,r_init):
#    LongTerm = -k*D/(2*(r_0-k*D*t/(2*r_0)-k/(np.sqrt(D*t/pi))))
#    ShortTerm = -k/2*np.sqrt(D/(pi*t))   
#    if LongTerm > ShortTerm*10:
#       return np.sqrt(1-k*D*t/B_init**2) # Long time solution
    
#   return (r_init-k*D*t/(2*r_init)-k*np.sqrt((D*t)/pi))/r_0 # Short time solution
    R = np.sqrt(r_init**2-k*D*t)/r_0
    if (R<0): R=0
    return R
   

def NextRLong(k_temp, t_temp, dt_temp, D_temp, r_init, r_prev):
    if k_temp*D_temp*t_temp > r_0**2: return 0
    r_temp = r_prev - (dt_temp*k_temp/2*(D_temp/r_prev + np.sqrt(D_temp/(pi*t_temp))))
    return r_temp

#Isokinetical solution, Normalised Volume Fraction of Spherical Particle for Two-Step annealing    
def VolFrac(t_temp, t_s, t_star_prev_temp, t_star_temp):
    VF_temp = (1-(t_s/t_star_prev_temp+(t_temp-t_s)/t_star_temp))**(3/2)
    if VF_temp < 0: return 0
    return VF_temp
    
def VolFrac1(k_temp,t_temp,D_temp,r_init):
    if k_temp*D_temp*t_temp/r_init**2>1: return 0
    return (1.0 - k_temp*D_temp*t_temp/r_init**2)**(3.0/2.0)  
    
# Calculates time to reach complete dissolution of particle
def t_star_f(r_temp, k_temp, D_temp):
    return r_temp**2/D_temp/k_temp
    
# Calculates analytical concentration profile
def CAnal(r,R,T,D,t,C_i_T):
    if((r-dx/2) <= R):
        return C_p
    return C_0+(C_i_T-C_0)*(R/r)*scipy.special.erfc((r-R)/(2.0*np.sqrt(D*t)))

 # Plots the analytical solution with diffusivity D(T_i)
def AnalConc():
    C_i_T = C_i_f(T_i)
    D_i = Diffusivity(T_i)
    Conc = [CAnal(i,r_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
    plt.figure(figsize=(14,10), dpi=120)
    plt.plot(x_bar,Conc)
    plt.xlim(0, L)
    plt.ylim(0.0, 1.0)
    plt.plot((0.0,t_i), (C_i_T, C_i_T), 'k-') # (x0,x1)(y0,y1) Visualisation of eq. particle/matrix concentration
    plt.xlabel('r [um]')
    plt.ylabel('Concentration of Si [wt.%]')
    plt.title('3D Analytic concentration profile of Si after %d seconds annealing at %d K' % (t_i, T_i))
    #plt.rcParams.update({'font.size': 18})
    return Conc
    
# Create diagonal and sub/super diagonal for tridiagonal sparse matrix, 3D case
def createSparse(DTemp1, D, R_temp):
    if R_temp == 0: R_temp = 1e-8 #Avoid divide by 0
    sup = [alpha*DTemp1/D*(1+(1/(i+R_temp/dx))) for i in range(N)] # sub and super diag. is non-equivalent for this finite difference scheme
    sub = [alpha*DTemp1/D*(1-(1/(1+i+R_temp/dx))) for i in range(N)] 
    diag = np.zeros(N+1)+(1-2*alpha*DTemp1/D) # diagonal
    return scipy.sparse.diags(np.array([sub,diag,sup]), [-1,0,1])
    
# Calculation of new concentration profile per time increment (sparse matrix) 
def nextTimeSparse(CVecT, ASparseT):
    return ASparseT*CVecT
    
# Iterative brute force method    
def nextTimeBruteForce(U_prev,R_n):
    U_temp = np.zeros(N+1)
    for i in range(N):
        if i==0: continue
        U_temp[i] = U_prev[i-1]*alpha*(1-1/(i+R_n/dx)) + U_prev[i]*(1.0-2.0*alpha)  + U_prev[i+1]*alpha*(1+1/(i+R_n/dx))
    return U_temp
    
# Calculation of concentration gradient at particle/matrix interface
def C_grad_interface(C_next,C_i_temp):
    return (C_next-C_i_temp)/dx
    
def NextR(D_temp, dt_temp, C_i_temp, r_prev,C_grad_interface_temp):
    r_temp = r_prev + dt_temp*D_temp/(C_p-C_i_temp)*C_grad_interface_temp
    if r_temp < 0:
        return 0
    return r_temp

def fin_diff(T1,T2,RSR_ch=-1): #RSR_ch is optinal
    if T1==T2:
        ShouldChange = False
    else:
        ShouldChange = True
        
    # Diffusion coefficients at temperatures T1 and T2
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    print('Diffusion coefficients D1(T1) and D2(T2):\n {0:.3e} um^2/s and {1:.3e} um^2/s\n'.format(D_1,D_2))
    D_max = max(D_1,D_2)

    # Dissolution at T = T1
    # Initialisation of parameters
    D_RSR = D_1
    R_RSR = r_0
    T_RSR = T1
    i_time = 0
    C_i_RSR = C_i_f(T_RSR)
    k_RSR = k_f(C_i_RSR)   
    t_star = t_star_f(r_0, k_RSR, D_RSR)
    t_star_prev = 1.0 # value currently irrelevant
    t_switch = 0
    print('D_RSR: {0:.3e}'.format(D_RSR))
    print('R_RSR: {0:.3e}'.format(R_RSR))
    print('T_RSR: {0:.3e}'.format(T_RSR))
    print('C_i_RSR: {0:.3e}'.format(C_i_RSR))
    print('k_RSR: {0:.3e}\n'.format(k_RSR))


    # Temporal discretisation
    print('Temporal discretisation:\n')
    dt = alpha*dx**2/D_max # D_max imposes the lowest timestep dt --> used to respect the stability criterion
    print('Time increments dt: {0:.3e} s'.format(dt))    
    Nt = math.ceil(t_i/dt)
    print('Time steps: %d\n' % Nt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
    # Creating initial concentration vector
#   index_cutoff = round(r_0/dx)
    index_cutoff = 0 # We define r = R_n + i*dx
#   U = np.append(np.zeros(index_cutoff)+C_p,np.zeros(N-index_cutoff+1)+C_0)
    U = np.zeros(N+1)+C_0
    print('Index cut-off: %d\n' % index_cutoff)
    U[index_cutoff] = C_i_RSR
    
#   print('Printing concentration first few elements\n')
#   print(U[:10:1])
#   print('Printing x_bar first few elements\n')
#   print(x_bar[:10:1])
#   print(U)
    
    # Create diag, sub and super diag for tridiag
    Sparse = createSparse(D_1,D_max,r_0) # must change for every new R_n
    #Solve for every timestep
    RSR_num = np.zeros(np.size(t))
    RSR_num[0] = r_0
    RSR_anal = np.zeros(np.size(t))
    VF_num = np.zeros(np.size(t))
    VF_num[0] = 1.0
    VF_isokin = np.zeros(np.size(t))
    VF_isokin[0] = 1.0    
    RVF_num_long = np.zeros(np.size(t))
    RVF_num_long[0] = r_0
    VF_num_long = np.zeros(np.size(t))
    VF_num_long[0] = 1.0
    
    for i in range(Nt):
        # Solve for every timestep
        C_grad_interface_temp = C_grad_interface(U[index_cutoff+1],U[index_cutoff])
        # U = nextTimeBruteForce(U,RSR_num[i])
        #U = nextTimeBruteForce(U,RSR_num[i])
        U = nextTimeSparse(U, Sparse)
        
        # Insert boundary conditions  
        # U[0:index_cutoff] = C_p
        U[index_cutoff] = C_i_RSR
        U[N] = C_0
        #VF_iso_temp = VolFrac1(k_RSR,i*dt,D_RSR,r_0)
        if not i==0:
            RVF_num_long[i] = NextRLong(k_RSR, i*dt, dt, D_RSR, r_0, RVF_num_long[i-1])
            VF_num_long[i] = (RVF_num_long[i]/r_0)**3

        VF_iso_temp = VolFrac(dt*i,t_switch, t_star_prev,t_star)
        
        RSR_anal[i] = R_f(k_RSR,dt*i_time,D_RSR,R_RSR) # !
        
        
#        if i == 9000: exit()
#        if i%500 == 0: print(C_grad_interface_temp)
#        VF_iso_temp = VolFrac(k_RSR,dt*i_time,D_RSR,R_RSR)
 #       if(i==1000 and False):
 #           #break
 #           print('Printing concentration first few elements after %d iterations\n' % i)
 #           print(U[:10:1])
            
        if ((RSR_num[i]/r_0)**3 < RSR_ch and ShouldChange):
        #if (VF_iso_temp < RSR_ch and ShouldChange):
            print('T1 and T2 are switched')
            # Redefine parameters at T2
            D_RSR = D_2
            T_RSR = T2
            C_i_RSR = C_i_f(T_RSR)
            k_RSR = k_f(C_i_RSR)
            #R_RSR = RSR_num_temp
            t_switch = i*dt
            t_star_prev = t_star
            t_star = t_star_f(r_0, k_RSR, D_RSR)
            i_time = 0 # Fix for two-step
            ShouldChange = False
            #Sparse = createSparse(D_2,D_max)
        if (VF_iso_temp > 0):
            VF_isokin[i] = VF_iso_temp
            
 #       if (RSR_num_temp > 0):
            VF_num[i]= (RSR_num[i]/r_0)**3
        i_time = i_time +1
 #       if i%10000 == 0:# and i<10001:
 #           plt.plot(x_bar,U)
 #       if(i == Nt): break
        RSR_num[i+1] = NextR(D_RSR,dt,C_i_RSR,RSR_num[i],C_grad_interface_temp)
 #       print(RSR_num[i+1])
 #       if i == 10:exit()
        if i%upSparse == 0:
            Sparse=createSparse(D_RSR, D_max, RSR_num[i+1])
#        if i==0: print(alpha*U[0]-alpha*U[0]/(r_0/dx+1))
#        if i%1==0 and i < 100 and True:
#            print(U[0],U[1],U[2],U[3])
#        if i==1001: xit()
        
    #plt.figure(figsize=(14,10), dpi=120)
    #plt.plot(x_bar,U,label='Numerical')
    #plt.ylim(0.0,1.0)
    if False: #change to True for more plots    
        plt.figure(figsize=(14,10), dpi=120)    
        C_i_T = C_i_f(T_i)
        D_i = Diffusivity(T_i)
        Conc = [CAnal(i,r_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
        plt.plot(x_bar,Conc,label='Analytical')
        x_barV2 = np.linspace(r_0,L+r_0,N+1)
        plt.plot(x_barV2[::5],U[::5],'o',label='Numerical')
        plt.ylim(0,1.0)
        plt.xlim(0,L/2)
        plt.xlabel('r [um]')
        plt.ylabel('Concentration of Si [wt.%]')
        plt.title('Concentration profile near spherical Si-particle in Al-Si after isothermal annealing at {0:.0f} K'.format(T_i),y=1.02)
        plt.legend()
        plt.rcParams.update({'font.size': 18})  
    #plt.plot((0.0,L), (C_i_T, C_i_T), 'k-') # (x0,x1)(y0,y1)
    #plt.title('Concentration profile after %d seconds annealing at two-step %d K and %d K temperature change' % (t_i,T1,T2)) 
    if True:  #Change to True for more plots.
        plt.figure(figsize=(14,10), dpi=120)
        plt.plot(t,(RSR_num/r_0)**3,'b-',label='Numerical')
        #plt.plot(t,(RSR_anal)**3,'r-',label='Isokinetic')
        plt.plot(t,VF_isokin,'r-',label='Isokinetic')
        #plt.ylim(0,1.0)
        #plt.xlim(0,t_i)
        plt.xlabel('Time [s]')
        plt.ylabel('Scaled volume fraction V/V_0')
        plt.title('Scaled volume fraction after two-step annealing at %d K and %d K' % (T1,T2),y=1.03)
        plt.legend()
        plt.rcParams.update({'font.size': 18})    
    if False: # Change to True for more plots
        plt.figure(figsize=(14,10), dpi=120)
        #plt.plot(t,VF_num,'b-',label='Numerical')
        plt.plot(t,RVF_num_long/r_0,'b-',label='Isothermal-Numerical')
        #plt.plot(t,VF_isokin,'r-',label='Isokinetic')
        plt.plot(t,(RSR_anal),'r-',label='Analytical-Short times')
        plt.ylim(0,1.0)
        plt.xlim(0,t_i)
        plt.xlabel('Time [s]')
        plt.ylabel('Normalised particle radius r/r_0')
        plt.title('Normalised particle radius after %d seconds annealing at %d K' % (t_i,T1))
        plt.legend()
        plt.rcParams.update({'font.size': 18})    
    
        plt.figure(figsize=(14,10), dpi=120)
        #plt.plot(t,VF_num,'b-',label='Numerical')
        plt.plot(t,VF_num_long,'b-',label='Isothermal-Numerical')
        #plt.plot(t,VF_isokin,'r-',label='Isokinetic')
        plt.plot(t,(RSR_anal)**3,'r-',label='Analytical-Short times')
        plt.ylim(0,1.0)
        plt.xlim(0,t_i)
        plt.xlabel('Time [s]')
        plt.ylabel('Scaled volume fraction V/V_0')
        plt.title('Scaled volume fraction after %d seconds annealing at %d K' % (t_i,T1))
        plt.legend()
        plt.rcParams.update({'font.size': 18})
    
        #plt.figure()
        #plt.plot(x_bar[index_cutoff::],U[index_cutoff::])
        #plt.ylim(-1.1,1.1)
        #plt.figure()
        #plt.plot(t,RSR_num)
    
def main(argv):
#    plt.figure(figsize=(14,10), dpi=120)

    #analytical = AnalConc() # Calc and plot concentration profiles, analytical formula
 #   finite_diff() # Calc and plot concentration profiles, finite differences
    #Plate_thickness()    
    fin_diff(T_hi,T_low,0.3)
 #   fin_diff(T_hi,T_low)
#    fin_diff(T_hi,T_low,0.7)
 #   NextBnum()
    plt.show()
#    fin_diff_wLin_Temp_profile_Cu() # Calc and plot concentration profile for Cu, linear temp. increase
#    stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
#    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main(sys.argv[1:])
