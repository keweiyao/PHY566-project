import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

#____A customized guassian fit function with parameter: width->sigma, x0->center, C->normlization_________
def gaussian(x, sigma, x0, C):
    return C/sigma*np.exp(-0.5*(x-x0)**2/sigma**2)

#__________Solver of diffusion equation with input:_________________________
#          initial condition, diffusion constant, time step, spatial step, maximum time of evolution
def solver(f0,                D,                  dt,        dx,           t_max):

    #Check the stability condition, if not satisfied, print information return with error value
    if dt>dx**2/2/D:
        print "time step too large, solution unstable"
        return -1, -1, -1
    
    #If dt, dx pass the stability test, assign initialize the memory space
    N_x = len(f0)           #Number of spaital grids
    N_t = int(t_max/dt)     #Number of time steps
    f = np.zeros([N_t, N_x])#Memory space
    f[0] = f0               #Initialization

    #Time evolution
    alpha = D*dt/dx**2
    for i in range(1, N_t):
        #Using perodic boundary condition
        for j in range(N_x):
            f[i][j] = (1.-2.*alpha)*f[i-1][j] + alpha*(f[i-1][(j-1)%N_x] + f[i-1][(j+1)%N_x])

    #return result, number of time steps and number of spatial grids
    return f, N_t, N_x

#_____________MAIN______________________________________
x = np.linspace(-8,8,500)   #x coordinates arrat
F0 = np.abs(x)<0.2          #delta function like initial condition
Dt = 0.0001                 #time step

#Calling Solver
F, n_t, n_x = solver(F0, 2., Dt, (np.max(x)-np.min(x))*1./(len(x)-1), 1.0)

#if the stability test is passed:
if n_t>0:
    #A little animiation of the time evolution
    plt.figure(figsize = (7,7))
    for i in range(n_t):
        if i%100==0:
            plt.clf()
            plt.plot(x, F[i], color = [i*1./n_t,0.3,1.0 - i*1./n_t], linewidth = 3., label = "$t = %f$"%(Dt*i), alpha = 0.7)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend(loc = "upper right")
            plt.axis([-5,5,0,np.max(F[0])])
            plt.pause(0.01)

    x_sample = np.linspace(-8,8,71)
    plt.figure(figsize = (7,7))
    #plot initial condition
    plt.plot(x,F0, 'g--', linewidth = 3, label = "Initial condition")

    #pick five snapshots of the evolution
    for i in [int(0.2*n_t),int(0.4*n_t),int(0.6*n_t),int(0.8*n_t),int(1.0*n_t)-1]:
        #plot them
        plt.plot(x, F[i], '-', color = [i*1./n_t,0.3,1.0 - i*1./n_t], linewidth = 3, label = "$t = %.2f$"%(Dt*i), alpha = 0.7)
        #fit them by gaussian wavepacket
        init_param = [1,0,1]
        best_param, cov = curve_fit(gaussian, x, F[i], init_param)
        #plot the fittted function and display the extracted variance of the density profile
        plt.plot(x_sample, gaussian(x_sample, best_param[0], best_param[1], best_param[2]), 'o', color = [i*1./n_t,0.3,1.0 - i*1./n_t], label = "$Fitted, \sigma^2(t)/t = %.2f$"%(best_param[0]**2/Dt/i), alpha = 0.7)
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("y", fontsize = 20)
    plt.legend(loc = "upper right", fontsize = 15)
    plt.axis([-8,8,0,np.max(F0)*0.25])
    plt.show()