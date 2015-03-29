import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

def gaussian(x, sigma, x0, C):
    return C/sigma*np.exp(-0.5*(x-x0)**2/sigma**2)

def solver(f0, D, dt, dx, t_max):

    if dt>dx**2/2/D:
        print "time step too large, solution unstable"
        return -1, -1, -1
    
    N_x = len(f0)
    N_t = int(t_max/dt)
    f = np.zeros([N_t, N_x])

    f[0] = f0

    for i in range(1, N_t):
        #Using perodic boundary condition
        for j in range(N_x):
            f[i][j] = f[i-1][j] + D*dt/dx**2*(f[i-1][(j-1)%N_x] + f[i-1][(j+1)%N_x] - 2.*f[i-1][j])

    return f, N_t, N_x

x = np.linspace(-10,10,500)
F0 = np.abs(x)<0.3

F, n_t, n_x = solver(F0, 2, 0.05, 0.5, 10)
if n_t>0:
    plt.figure(figsize = (7,7))
    for i in range(n_t):
        plt.clf()
        plt.plot(x, F[i], color = [i*1./n_t,0.3,1.0 - i*1./n_t], linewidth = 3., label = "t = %f"%(0.05*i), alpha = 0.7)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc = "upper right")
        plt.axis([-5,5,0,np.max(F[0])])
        plt.pause(0.01)
    
    x_sample = np.linspace(-3,3,21)
    plt.figure(figsize = (7,7))
    for i in [int(0.2*n_t),int(0.5*n_t),int(0.8*n_t)]:
        plt.plot(x, F[i], '-', color = [i*1./n_t,0.3,1.0 - i*1./n_t], linewidth = 3, label = "t = %f"%(0.05*i), alpha = 0.7)
        init_param = [1,0,1]
        best_param, cov = curve_fit(gaussian, x, F[i], init_param)
        plt.plot(x_sample, gaussian(x_sample, best_param[0], best_param[1], best_param[2]), 'o', color = [i*1./n_t,0.3,1.0 - i*1./n_t], label = "t = %f, fitted"%(0.05*i), alpha = 0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc = "upper right")
    plt.axis([-5,5,0,np.max(F[0])])
    plt.show()