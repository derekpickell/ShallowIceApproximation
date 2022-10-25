# Shallow Ice Approximation 
# Derek Pickell
# 10/21/22 
# Adaptive explicit method, conditionally stable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def diffusion(grid_x, grid_y, length, width, Dup, Ddown, Dleft, Dright, u, tf):
    dx = 2 * length / grid_x # grid unit size (km) ie 10km/10 = 1km units       
    dy = 2 * width / grid_y

    # F = np.zeros((int(length/dx), int(width/dy)))
    b = np.zeros(np.shape(u))

    t = 0
    k = 0
    while t < tf:
        # stability condition gives time-step restriction
        maxD = [Dup.max(), Ddown.max(), Dleft.max(), Dright.max()]
        maxD = max(maxD)

        if maxD <= 0.0:
            dt = tf - t 

        else:
            dt0 = 0.25 * min(dx, dy)**2 / maxD
            dt = min(dt0, tf - t )

        gamma_x = dt / (dx ** 2)
        gamma_y = dt / (dy ** 2)

        Ub = np.add(u, b)
        A = Ub[2:  , 1:-1] # u_{i+1,j}^k
        B = Ub[ :-2, 1:-1] # u_{i-1,j}^k
        C = Ub[1:-1,  :-2] # u_{i,j-1}^k 
        D = Ub[1:-1,   2:] # u_{i,j+1}^k
        E = Ub[1:-1, 1:-1] # u_{i,j}^k

        u[1:-1, 1:-1] = E +  gamma_y * Dup * (D - E) - \
                             gamma_y * Ddown * (E - C) + \
                             gamma_x * Dright * (A - E) - \
                             gamma_x * Dleft * (E - B) 

        # u = u + F * dt
        t = t + dt
        # print("time", t)
        # print("tf", tf)

        k +=1 
        # print("k", k)
    return u, (tf/k)
        
def siaflat(grid_x, grid_y, length, width, H0, delta_t, tf):
    g = 9.8                 # m/s^2
    rho = 910.0             # kg/m^3
    A = 3.16e-24            # ice flow parameter
    const = 2 * A * (rho * g)**3 / 5 
    
    dx = 2 * length / grid_x # grid unit size (km) ie 10km/10 = 1km units       
    dy = 2 * width / grid_y

    t = 0
    dtlist = []
    N = np.ceil(tf / delta_t) # number of iterations, based on time step
    deltat = tf/N             # adjusted time step for each iteration

    # fig, ax = plt.subplots(figsize=(8,6))
    # fig.suptitle("Ice Elevation")
    # fig.tight_layout()

    xx, yy = np.meshgrid(np.arange(0, grid_x, 1), np.arange(0, grid_y, 1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle("Ice")

    H = H0
    
    for n in range(0, int(N)):
        # staggered grid thicknesses
        Hup = 0.5 * ( H[1:grid_x-1, 2:grid_y] + H[1:grid_x-1, 1:grid_y-1] ) # up and down
        Hdn = 0.5 * ( H[1:grid_x-1, 1:grid_y-1] + H[1:grid_x-1, :grid_y-2] )
        Hrt = 0.5 * ( H[2:grid_x, 1:grid_y-1] + H[1:grid_x-1, 1:grid_y-1] ) # right and left
        Hlt = 0.5 * ( H[1:grid_x-1, 1:grid_y-1] + H[:grid_x-2, 1:grid_y-1] )

        # staggered grid value of |grad h|^2 = alpha^2
        a2up = (H[2:grid_x, 2:grid_y] + H[2:grid_x, 1:grid_y-1] - H[:grid_x-2, 2:grid_y] - H[:grid_x-2, 1:grid_y-1])**2 / (4*dx)**2 + \
                (H[1:grid_x-1, 2:grid_y] - H[1:grid_x-1, 1:grid_y-1])**2 /dy**2
        a2dn = (H[2:grid_x, 1:grid_y-1] + H[2:grid_x,:grid_y-2] - H[:grid_x-2,1:grid_y-1] - H[:grid_x-2,:grid_y-2])**2 / (4*dx)**2 + \
                (H[1:grid_x-1,1:grid_y-1] - H[1:grid_x-1,:grid_y-2])**2 /dy**2
        a2rt = (H[2:grid_x,1:grid_y-1] - H[1:grid_x-1,1:grid_y-1])**2 / dx**2 + \
                (H[2:grid_x,2:grid_y] + H[1:grid_x-1,2:grid_y] - H[2:grid_x,:grid_y-2] - H[1:grid_x-1,:grid_y-2])**2 / (4*dy)**2
        a2lt = (H[1:grid_x-1,1:grid_y-1] - H[:grid_x-2,1:grid_y-1])**2 / dx**2 + \
                (H[:grid_x-2,2:grid_y] + H[1:grid_x-1,2:grid_y] - H[:grid_x-2,:grid_y-2] - H[1:grid_x-1,:grid_y-2])**2 / (4*dy)**2

        # Grid diffusivity D = const H^{n+2} |grad h|^{n-1}, Mahaffy evaluation of staggered grid diffusivity
        Dup = const * Hup**5 * a2up
        Ddn = const * Hdn**5 * a2dn
        Drt = const * Hrt**5 * a2rt
        Dlt = const * Hlt**5 * a2lt

        H, dtadapt = diffusion(grid_x, grid_y, length, width, Dup, Ddn, Drt, Dlt, H, deltat)

        # ax.clear()
        # ax.imshow(H, interpolation=None, cmap=plt.cm.jet)
        # plt.pause(.1)

        ax.clear()
        ax.set_xlabel(n)
        surf = ax.plot_surface(xx, yy, H, cmap=cm.jet, linewidth=0, antialiased=False)
        plt.pause(.1)

        t = t + deltat
        dtlist.append(dtadapt)

    return [H, dtlist]

def roughice():
    length = 500e3           # kilometers, half length
    width = length
    grid_x = 60             # grid size (unitless, keep small for computation time)
    grid_y = grid_x
    dx = 2 * length / grid_x # grid unit size (km) ie 10km/10 = 1km units       
    dy = 2 * width / grid_y
    delta_t = 31556926 * .2  # seconds
    final_time = 31556926*50 # seconds

    u0 = np.random.rand(grid_x, grid_y) * 1000 + 3000 # initialize surface heights
    # u0 = np.full((grid_x, grid_y), 2000)

    # rough up the ice
    for i in range(0, grid_x):
        for j in range(0, grid_y):
            if ((i-grid_x/2)**2 + (j-grid_y/2)**2 > (2*(.25*grid_x)**2)):
                u0[i, j] = 0

    xx, yy = np.meshgrid(np.arange(0, grid_x, 1), np.arange(0, grid_y, 1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle("Ice")
    surf = ax.plot_surface(xx, yy, u0, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # get heights and times (RUN)
    [H, dtlist] = siaflat(grid_x, grid_y, length, width, u0, delta_t, final_time)


if __name__=="__main__":

    roughice()  