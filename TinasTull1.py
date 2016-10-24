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
R = 8.31446 # [J/(K*mol)] Univ. Gas Constant  
NA = 6.022*10**23 # Avogadro's constant, [particles/mol]
"Isothermal annealing at T = 400 [C]"
T_K = 273.15 # Deg K at 0 deg C
T_i = 400.0+T_K # [K]
T_low = 400.0+T_K # [K]
T_hi = 430.0+T_K # [K]

"From table 1 in Bjørneklett"
#C_star = 2.15e3 # [wt.%]
C_star = 2.17e3 # [wt.%]
DeltaH = 50.8e3 # [J/mol]
D_0 = 3.46e7 # [um^2*s^-1] Diffusion coeffisient
Q = 123.8e3 # [J/mol]
B_0 = 1.0e-3 # [um] Initial half thickness of plate-shaped particle
C_p = 1.0e2 # [wt.%] Particle concentration of Si
C_0 = 0.0  # [wt.%] Matrix concentration of Si

#Diffusivity at T_i
D_i = D_0*np.exp(-Q/(R*T_i))
print('Diffusion coefficient at {0:.0f} K: {1:.3e} um^2/s\n'.format(T_i,D_i))

"Spatial discretisation"
print('Spatial discretisation:')
N = 500 # Number of spatial partitions of bar
L = 1.5 # [um] Length of bar 
t_i = 15 # Seconds for isothermal annealing
x_bar = np.linspace(0,L,N+1)
dx = L/N   # Need N+1 x points, where [N/2] is centered in a 0-indexed array
print('Bar partitioning dx: {0:.3e} um\n'.format(dx))

# The stability criterion for the explicit finite difference scheme must be fulfilled
alpha = .4  # alpha = D*dt/dx**2 --> Const in discretisation --> Must be <= 0.5 if used to find dt


"Precipitation of pure Si particles in a binary Al-Si alloy, assuming a diluted Al matrix"
# Calculating and plotting concentration profile for the spatial range [0,L] um after t_i s annealing at T_i deg C

C = C_star*np.exp(-DeltaH/(R*T_i)) # Interface concentration at temp T_i
print('Interface concentration: {0:.3e} wt. percentage\n'.format(C))
# Calculates concentration [wt.%] at particle interface at temperature Ttemp [K]
def C_i_f(Ttemp):
    return C_star*np.exp(-DeltaH/(R*(Ttemp)))

# Calculates diffusivity at temp. T [K]
def Diffusivity(T):
    return D_0*np.exp(-Q/(R*T))

# Calculation of concentration ratio coefficient k
def k_f(C_it):
    #return 2*(C_it-C_0)/(C_p-C_0) # Paper Bjørneklett et al definition
    return 2*(C_it-C_0)/(C_p-C_it) # Whelan definition

# Analytical calculation of plate half thickness, iso-thermal annealing
def Bf(k_temp,t_temp,D_temp,B_init):
    return (B_init-k_temp*(np.sqrt(D_temp*t_temp/pi))) # Task 2A (b) eq. 10, Bjørneklett et al, isothermal
def Bf2(k_temp,t_temp,dt_temp,D_temp,B_prev):
    return B_prev-dt_temp*k_temp*np.sqrt(D_temp/(pi*t_temp))/2 # Task 2A (c) eq. 9, Bjørneklett et al, isokinetic
    
T_dis = pi/D_i*B_0**2/(k_f(C_i_f(T_i)))**2
print('Time to completely dissolve the particle at T_ref = T_i = {0:.0f} K: {1:.0f} s\n' .format(T_i,T_dis))

# Analytical calculation of concentration profile [wt.%], iso-thermal annealing
def C_Analytical(x,r,T,D,t,C_i_T):
    if((x-dx/2) <= r):
        return C_p
    return (C_i_T-C_0)*scipy.special.erfc((x-r)/(2.0*np.sqrt(D*t)))

# Plotting the analytical solution with diffusivity D = D(T) (independent of x), iso-thermal annealing
def PlotAnalyticalC():
    C_i_T = C_i_f(T_i) # Particle interface concentration at temp. T_i [K]
    Conc_1 = [C_Analytical(i,B_0,T_i,D_i,t_i/4,C_i_T) for i in x_bar]
    Conc_2 = [C_Analytical(i,B_0,T_i,D_i,t_i/2,C_i_T) for i in x_bar]
    Conc_3 = [C_Analytical(i,B_0,T_i,D_i,t_i*3/4,C_i_T) for i in x_bar]
    Conc_4 = [C_Analytical(i,B_0,T_i,D_i,t_i,C_i_T) for i in x_bar]
    plt.figure(figsize=(14,10), dpi=120)
    plt.plot(x_bar,Conc_1,'-', label='Analytical {:.2f}s'.format(t_i/4))
    plt.plot(x_bar,Conc_2,'-', label='Analytical {:.2f}s'.format(t_i/2))
    plt.plot(x_bar,Conc_3,'-', label='Analytical {:.2f}s'.format(t_i*3/4))
    plt.plot(x_bar,Conc_4,'-', label='Analytical {:.2f}s'.format(t_i))
    plt.plot((B_0, B_0), (C_i_T, C_p), 'k-') # Vertical line to indicate particle->interface Si [wt.%] concentration drop
    plt.xlim(0, L)
    plt.ylim(0, 0.3)
    plt.xlabel('x [um]')
    plt.ylabel('Si concentration [wt.%]')
    plt.title('Concentration profile near plate-shaped Si-particle in Al-Si after isothermal annealing at {0:.0f} K'.format(T_i),y=1.02)
 #   plt.legend(bbox_to_anchor=(0.2,1))
    plt.legend()
    plt.rcParams.update({'font.size': 16})
    return Conc_4
    
# Creating diagonal and sub/super diagonal for tri-diagonal sparse matrix
# Finite Difference Scheme: Forward time, Central space applied to Fick's 2nd law: dC/dt = Dd^2C/dx^2
# alpha = delta_t*D/delta_x^2
def createSparse(D, D_max):
    subsup = np.zeros(N)+alpha*D/D_max      # sub and super diagonal is equivalent for this finite difference scheme
    diag = np.zeros(N+1)+1-2*alpha*D/D_max  # diagonal elements
    return scipy.sparse.diags(np.array([subsup,diag,subsup]), [-1,0,1])

# Calculation of new concentration profile per time increment (sparse matrix implementation) 
def nextTimeSparse(CVecT, ASparseT):
    return CVecT*ASparseT
    
# Calculation of plots for animation
def saveFig(xVecT,CVecT,timeT,figNameT):
    plt.plot(xVecT, CVecT, label='Cu')
    plt.xlim(0,L)
    plt.ylim(0, 1.0)
    plt.xlabel('x [um]')
    plt.ylabel('Concentration Si [wt.%]')
    plt.title('Concentration profile of Si in Al-Si after {0:.0f} hours annealing at {1:.0f} K' % (timeT, T_anneal))
    plt.legend(bbox_to_anchor=(0.2,1))
    plt.rcParams.update({'font.size': 18})
    plt.savefig(figNameT,transparant=True)

# Calculation of concentration gradient at particle/matrix interface
def C_grad_interface(C_next,C_i_temp):
    return (C_next-C_i_temp)/dx
    
# Calculation of new particle plate half thickness from mass conservation
def NextB(D_temp, dt_temp,C_i_temp, B_prev,C_grad_interface_temp):
    B = B_prev + dt_temp*D_temp/(C_p-C_i_temp)*C_grad_interface_temp
    if B < 0:
        return 0
    return B
    
# Isokinetic calculation of plate half thickness, non-isothermal annealing
def NextBrel(t_temp, t_s, t_star_prev_temp, t_star_temp):
    B = 1-np.sqrt(t_s/t_star_prev_temp+(t_temp-t_s)/t_star_temp)
    if B < 0:
        return 0
    return B

# Calculates total time for complete dissolution
def t_star_f(B_temp, k_temp, D_temp):
    return pi/D_temp*(B_temp/k_temp)**2

# Finite difference solution of Si-particle dissolution/growth in Al-Si when exposed to a two-step temperature 
# shift using sparse matrices 
# For isothermal case set T1 = T2
def fin_diff(T1,T2,RPT_ch):
    if T1==T2:
        ShouldChange = False # Boolean variable which determines if a two-step temperature shift should be applied
    else:
        ShouldChange = True
        
    # Diffusion coefficients at temperatures T1 and T2
    D_1 = Diffusivity(T1)
    D_2 = Diffusivity(T2)
    print('Diffusion coefficients D1(T1) and D2(T2):\n {0:.3e} um^2/s and {1:.3e} um^2/s\n'.format(D_1,D_2))
    D_max = max(D_1,D_2)
    
    # Temporal discretisation
    print('Temporal discretisation:\n')
    dt = alpha*dx**2/D_max # D_max imposes the lowest timestep dt --> used to respect the stability criterion
    print('Time increments dt: {0:.3e} s'.format(dt))    
    Nt = math.ceil(t_i/dt)
    print('Time steps: %d\n' % Nt)
    t = np.linspace(0, t_i, Nt+1) # Mesh points in time
    
    # Creating initial concentration vector
    index_cutoff = round(B_0/dx) # index of particle/matrix interface
    print('Index of particle/matrix interface: {0:.0f}\n'.format(index_cutoff))
    U = np.append(np.zeros(index_cutoff)+C_p,np.zeros(N-index_cutoff+1))
    
    # Creating sparse matrix
    Sparse = createSparse(D_1,D_max)

    # Initialisation of solution vectors, RPT: Relative Plate Thickness
    RPT_isokin = np.zeros(np.size(t))
    RPT_Num_Iso = np.zeros(np.size(t))
    RPT_Iso_Ana = np.zeros(np.size(t))
    RPT_num = np.zeros(np.size(t))
    
    # Dissolution at T = T1
    # Initialisation of parameters
    RPT_num[0] = B_0 # Numerical solution, mass balance
    RPT_Num_Iso[0] = B_0 #<---- Task 2A (c) Numerical Isokinetic
    RPT_Iso_Ana[0] = B_0 #<---- Task 2A (b) Isothermal Annealing
    RPT_isokin[0] = B_0 #<---- Isokinetic Solution
    D_RPT = D_1
    B_RPT = B_0
    T_RPT = T1
    i_time = 0
    C_i_RPT = C_i_f(T_RPT)
    k_RPT = k_f(C_i_RPT)
    t_switch = 0.0
    t_star = t_star_f(B_RPT, k_RPT, D_RPT) # Time to reach complete particle dissolution
    t_star_prev = 1.0   # Value is currently irrelevant
    RPT_temp_Num_Iso = B_0
    U[index_cutoff]=C_i_RPT
    # Option to plot several concentration profiles after temperature switch
    P = False
    PlotSeveral = False
    # Option to compare numerical and analytic concentration profiles at several isothermal annealing times
    CompAandN = False
    for i in range(Nt):
        # Solve for every timestep
        U = nextTimeSparse(U, Sparse)
        # Insert boundary conditions
        U[index_cutoff] = C_i_RPT
        U[N] = 0 # infinite BC
        RPT_temp_Iso_Ana = Bf(k_RPT,dt*i_time,D_RPT,B_RPT)            #<---- Task 2A (b) Isothermal Annealing
        
        if not i ==0: RPT_temp_Num_Iso = Bf2(k_RPT,dt*i_time,dt,D_RPT,RPT_Num_Iso[i-1])     #<---- Task 2A (c) Numerical Isokinetic      
        RPT_temp = NextBrel(dt*i,t_switch,t_star_prev,t_star) # Numerical solution, mass balance
        C_grad_interface_RPT = C_grad_interface(U[index_cutoff+1],U[index_cutoff]) # Input: concentration at index[particle/matrix boundary]+1 and index[particle/matrix boundary]
        
        if (RPT_temp < RPT_ch and ShouldChange):
            ShouldChange = False
            # Update parameters
            print('Previous diffusion coefficient: {0:.3e} um^2/s'.format(D_RPT))
            D_RPT = D_2
            print('Now changed to: {0:.3e} um^2/s\n'.format(D_RPT))
            B_RPT = RPT_temp*B_0 # New initial thickness
            print('Thickness at temperature switch: {0:.3e} um'.format(B_RPT))
            T_RPT = T2 # New temp
            C_i_RPT = C_i_f(T_RPT) # New particle/matrix interface concentration
            k_RPT = k_f(C_i_RPT) # New concentration ratio coefficient
            Sparse = createSparse(D_2,D_max)
            i_time = 0 # Reset
            t_switch = dt*i # Time at temperature switch
            t_star_prev = t_star
            t_star = t_star_f(B_0, k_RPT, D_RPT) # Updated at new temperature
            print('Dissolution times t_star_prev and t_star: {0:.3e} s and {1:.3e} s'.format(t_star_prev,t_star))
        if (RPT_temp > 0):
            RPT_isokin[i] = RPT_temp
            RPT_Iso_Ana[i] = RPT_temp_Iso_Ana
            RPT_Num_Iso[i] = RPT_temp_Num_Iso    
        if (i==Nt): continue
        RPT_num[i+1] = NextB(D_RPT,dt,C_i_RPT, RPT_num[i], C_grad_interface_RPT)
        if (i_time == 1):
            #plt.plot(x_bar,U) # At start and switch
            if PlotSeveral:
                P = True
                plt.figure(figsize=(14,10), dpi=120)
            PlotSeveral = False # NB
        if (P and i_time%100==0 and i_time < 10000 and False): #NB True
            plt.plot(x_bar,U)
            plt.xlabel('x [um]')
            plt.ylabel('Concentration Si [wt.%]')
            plt.title('Analytic concentration profile of Si [wt.%]')
        i_time = i_time+1
        
        if any(i*dt<t_i*itt+dt/2 and i*dt>t_i*itt-dt/2 for itt in [1/4,1/2,3/4,1]) and CompAandN: # Numerical comparison to analytical concentration profile
            #plt.figure(figsize=(14,10), dpi=120)
            plt.plot(x_bar[::5], U[::5],'o', label='Numerical {:.2f}s'.format(i*dt))
            #plt.plot((B_0, B_0), (C_i_RPT, C_p), 'k-') # (x0,x1)(y0,y1)
            #plt.xlim(0, L)
            #plt.ylim(0, 1.1)
            #plt.xlabel('x [um]')
            #plt.ylabel('Concentration Si [wt.%]')
            #plt.title('Analytic and numerical concentration profile of Si after isothermal annealing at %d K' % (T1))
            plt.legend()
            plt.rcParams.update({'font.size': 16})
        
    TwoStep = True
    Single = False
    if TwoStep:
        RPT_num = [i/B_0 for i in RPT_num]
        RPT_isokin = [i for i in RPT_isokin]
        #plt.figure(figsize=(14,10), dpi=120)
        plt.plot(t,RPT_isokin,label='Isokinetic')
        plt.plot(t[::500],RPT_num[::500],'o',label='Numerical')
        plt.xlim(0, t_i)
        plt.ylim(0.0, 1.0)
        plt.xlabel('t [s]')
        plt.ylabel('Relative plate thickness B/B_0')
        plt.plot((0.0,t_switch), (RPT_ch, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.plot((t_switch,t_switch), (0.0, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.title('Relative plate thickness after a two-step annealing at %d K and %d K' % (T1,T2),y=1.02)
        plt.legend()
        plt.rcParams.update({'font.size': 18})
    elif Single:
        RPT_isokin = [i for i in RPT_isokin]
        plt.figure(figsize=(14,10), dpi=120)
        plt.plot(t,RPT_isokin,label='Isothermal')
        #plt.plot(t,RPT_num,label='Numerical')
        plt.xlim(0, t_i)
        plt.ylim(0.0, 1.0)
        plt.xlabel('t [s]')
        plt.ylabel('Relative plate thickness B/B_0')
        plt.plot((0.0,t_switch), (RPT_ch, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.plot((t_switch,t_switch), (0.0, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.title('Relative plate thickness after a two-step annealing at %d K and %d K' % (T1,T2),y=1.02)
        plt.legend()
        plt.rcParams.update({'font.size': 18})
        
        RPT_Iso_Ana = [i/B_0 for i in RPT_Iso_Ana]
        print(RPT_Num_Iso)
        RPT_Num_Iso = [i/B_0 for i in RPT_Num_Iso]
        plt.figure(figsize=(14,10), dpi=120)
        plt.plot(t,RPT_Iso_Ana,label='Analytical')
        plt.plot(t[::500],RPT_Num_Iso[::500],'o',label='Numerical Isokinetic')
        #plt.plot(t,RPT_num,label='Numerical')
        plt.xlim(0, t_i)
        plt.ylim(0.0, 1.0)
        plt.xlabel('t [s]')
        plt.ylabel('Relative plate thickness B/B_0')
        plt.plot((0.0,t_switch), (RPT_ch, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.plot((t_switch,t_switch), (0.0, RPT_ch), 'k-') # (x0,x1)(y0,y1)
        plt.title('Relative plate thickness after isothermal annealing at %d K' % (T1),y=1.02)
        plt.legend()
        plt.rcParams.update({'font.size': 18})
    else: single = False
    
def stabilityCheck(exact,approx):
    residual = np.zeros(len(exact))
    residual = np.log10(np.abs(exact-approx))
    #residual = [i/j for i,j in zip(residual,exact)]
    figureX = plt.figure(figsize=(14,10),dpi=600)
    plt.rcParams.update({'font.size': 18})
    plt.plot(np.linspace(-1, 1, N+1),residual,'r')
    plt.xlim(-1, 1)
    plt.xlabel('x [um]')
    plt.ylabel('Log10(residuals)')
    plt.title('Comparison between analytical and finite difference solutions')
    plt.rcParams.update({'font.size': 16})

def main(argv):
    plt.figure(figsize=(14,10), dpi=120)
    #PlotAnalyticalC() # Calculate and plot isokinetic concentration profiles, isothermal case
    #fin_diff(T_low,T_hi,0.3)
    #fin_diff(T_low,T_hi,0.7)
    fin_diff(T_hi,T_low,0.3)
    fin_diff(T_hi,T_low,0.7)
    plt.show()
    
    #stabilityCheck(analytical,fin_diff) # Comparison analytical and finite differences
    #print("--- %s seconds ---" % (time.time() - start_time)) <--- Timing of run-time
    
if __name__ == "__main__":
    main(sys.argv[1:])
