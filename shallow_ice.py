# Shallow Ice Approximation 
# Derek Pickell
# 11/4/22 
# Adaptive explicit method, conditionally stable
# Some of the methods used here adapted from https://glaciers.gi.alaska.edu/sites/default/files/notes-bueler-2016.pdf
# Some data adapted (BedMachine): Morlighem M. et al., (2017), BedMachine v3: Complete bed topography and ocean bathymetry mapping of Greenland from multi-beam echo sounding combined with mass conservation, Geophys. Res. Lett., 44, doi:10.1002/2017GL074954, http://onlinelibrary.wiley.com/doi/10.1002/2017GL074954/full

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

def diffusion(dx, dy, Dup, Ddown, Dleft, Dright, u, tf, b, M):

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
        A = Ub[2:  , 1:-1] # u_{i+1,j}
        B = Ub[ :-2, 1:-1] # u_{i-1,j}
        C = Ub[1:-1,  :-2] # u_{i,j-1} 
        D = Ub[1:-1,   2:] # u_{i,j+1}
        E = Ub[1:-1, 1:-1] # u_{i,j}
        F = u[1:-1, 1:-1] 

        u[1:-1, 1:-1] = F +  gamma_y * Dup * (D - E) - \
                             gamma_y * Ddown * (E - C) + \
                             gamma_x * Dright * (A - E) - \
                             gamma_x * Dleft * (E - B) 

        u = u + M * dt
        t = t + dt
        k +=1 
    return u, (tf/k)
        
def sia(nx, ny, H0, delta_t, tf, dx, dy, X, Y, b, M, A):
    """
    H = ice thickness
    D = [2 A (rho g)^3 / 5] H^5 |grad H|^2 = nonlinear diffusivity
    Mahaffy (1976) method to evaluate map-plane diffusivity
    """
    g = 9.8                 # m/s^2
    rho = 910.0             # kg/m^3
    const = 2 * A * (rho * g)**3 / 5 
    f = rho / 1028.0        # fraction floating ice below surface

    t = 0
    dtlist = []
    N = np.ceil(tf / delta_t) # number of iterations, based on time step
    deltat = tf/N             # adjusted time step for each iteration

    plot = False
    if plot == True: 
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.suptitle("Ice")

    H = np.where(H0 < 0, 0.0, H0)
    
    for n in range(0, int(N)):
        # staggered grid thicknesses
        h = H + b
        h = np.where(h<0.0, 0.0, h)

        Hup = 0.5 * ( H[1:nx-1, 2:ny] + H[1:nx-1, 1:ny-1] ) # up and down
        Hdn = 0.5 * ( H[1:nx-1, 1:ny-1] + H[1:nx-1, :ny-2] )
        Hrt = 0.5 * ( H[2:nx, 1:ny-1] + H[1:nx-1, 1:ny-1] ) # right and left
        Hlt = 0.5 * ( H[1:nx-1, 1:ny-1] + H[:nx-2, 1:ny-1] )

        # staggered grid value of |grad h|^2 = alpha^2
        a2up = (H[2:nx, 2:ny] + H[2:nx, 1:ny-1] - H[:nx-2, 2:ny] - H[:nx-2, 1:ny-1])**2 / (4*dx)**2 + \
                (H[1:nx-1, 2:ny] - H[1:nx-1, 1:ny-1])**2 /dy**2
        a2dn = (H[2:nx, 1:ny-1] + H[2:nx,:ny-2] - H[:nx-2,1:ny-1] - H[:nx-2,:ny-2])**2 / (4*dx)**2 + \
                (H[1:nx-1,1:ny-1] - H[1:nx-1,:ny-2])**2 /dy**2
        a2rt = (H[2:nx,1:ny-1] - H[1:nx-1,1:ny-1])**2 / dx**2 + \
                (H[2:nx,2:ny] + H[1:nx-1,2:ny] - H[2:nx,:ny-2] - H[1:nx-1,:ny-2])**2 / (4*dy)**2
        a2lt = (H[1:nx-1,1:ny-1] - H[:nx-2,1:ny-1])**2 / dx**2 + \
                (H[:nx-2,2:ny] + H[1:nx-1,2:ny] - H[:nx-2,:ny-2] - H[1:nx-1,:ny-2])**2 / (4*dy)**2

        # Grid diffusivity D = const H^{n+2} |grad h|^{n-1}, Mahaffy evaluation of staggered grid diffusivity
        Dup = const * Hup**5 * a2up
        Ddn = const * Hdn**5 * a2dn
        Drt = const * Hrt**5 * a2rt
        Dlt = const * Hlt**5 * a2lt

        H, dtadapt = diffusion(dx, dy, Dup, Ddn, Drt, Dlt, H, deltat, b, M)
        H = np.where(H<0.0, 0.0, H)
        # calvehere = (b < - f * H)

        if plot == True:
            ax.clear()
            print_time = int(deltat*n)/31556926
            ax.set_xlabel('time: %.3f years' % print_time)
            surf = ax.plot_surface(X, Y, H, cmap=cm.jet, linewidth=0, antialiased=False)
            plt.pause(.1)

        t = t + deltat
        dtlist.append(dtadapt)
    h = H + b
    h = np.where(h<0.0, 0.0, h)
    return [H, h, dtlist]

def roughice():
    # Ice softness
    A = 3.16e-24            # ice flow parameter 1e-16/seconds per year

    # Spatial and Temporal Parameters
    Lx = 500e3           # kilometers, half length, spatial extent
    Ly = Lx
    nx = 60             # number of points (unitless, keep small for computation time)
    ny = nx
    dx = 2 * Lx / nx # grid unit size (km) ie 10km/10 = 1km units       
    dy = 2 * Ly / ny
    x = np.linspace(-Lx,Lx,nx) 
    y = x
    X, Y = np.meshgrid(x,y)
    delta_t = 31556926 * .2  # seconds
    final_time = 31556926*50 # seconds (sec/year * number of years)

    # Initial Arrays
    u0 = np.random.rand(nx, ny) * 1000 + 3000 # initialize surface heights
    M = np.zeros(np.shape(u0))
    b = np.zeros(np.shape(u0))

    # rough up the ice
    for i in range(0, nx):
        for j in range(0, ny):
            if ((i-nx/2)**2 + (j-ny/2)**2 > (2*(.25*nx)**2)):
                u0[i, j] = 0

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.suptitle("Ice")
    surf = ax.plot_surface(X, Y, u0, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # get heights and times (RUN)
    [H, h, dtlist] = sia(nx, ny, u0, delta_t, final_time, dx, dy, X, Y, b, M, A)

def getCDF(scale_factor):
    """parse CDF package: 
    Morlighem M. et al., (2017), BedMachine v3: Complete bed topography and ocean bathymetry mapping of Greenland from multi-beam echo sounding combined with mass conservation, Geophys. Res. Lett., 44, doi:10.1002/2017GL074954, http://onlinelibrary.wiley.com/doi/10.1002/2017GL074954/full
    """
    filename = 'BedMachineGreenland-v5.nc'
    f = nc.Dataset(filename)
    x = f.variables['x'][:] # temperature variable
    y = f.variables['y'][:]
    X, Y = np.meshgrid(x[::scale_factor],y[::scale_factor])
    bed = f.variables['bed'][:]
    thickness = f.variables['thickness'][:]
    surface = f.variables['thickness'][:]
    f.close()
    bed_scaled = bed[1::scale_factor, 1::scale_factor]
    thickness_scaled = thickness[1::scale_factor, 1::scale_factor]
    surface_scaled = surface[1::scale_factor, 1::scale_factor]
    
    # Display bed elevation
    # fig, ax = plt.subplots()
    # plt.pcolor(X, Y, bed_scaled, cmap=plt.cm.jet)
    # plt.show()
    
    return X, Y, thickness_scaled, bed_scaled, thickness_scaled

def greenland():
    """
    Get Bed and Ice thickness from Bedmachine
    Run diffusion model 
    """
    grid_resolution = 400 # kilometers
    scale_factor = int(grid_resolution*100 / 150) # scale factor for bedmachine's 150m resolution

    X, Y, thickness_scaled, bed_scaled, thickness_scaled = getCDF(scale_factor)
    M = np.zeros(np.shape(bed_scaled))

    # Spatial and Temporal Parameters
    nx = int(np.shape(X)[0])             # number of points (unitless, keep small for computation time)
    ny = int(np.shape(X)[1])
    dx = grid_resolution*nx
    dy = grid_resolution*ny

    seconds_per_year = 31556926
    deltata = 1.0
    tblocka = 500.0 # time blocks in years
    N = 80  # number of blocks of length tblock
    # tfa = N * tblocka
    # t = np.linspace(0, N)
    # t = t * tblocka * seconds_per_year
    # print('doing run of %.3f ka total, in blocks of %.3f a,\n', tfa/1000.0,tblocka)

    # fix units and parameters for actual run
    delta_t = deltata * seconds_per_year  # convert to seconds
    final_time = tblocka * seconds_per_year

    A = 3 * 1.0e-16 / seconds_per_year

    initial_surf = getSurface(thickness_scaled, bed_scaled)
    
    H = thickness_scaled

    # fig, ax = plt.subplots(1)
    # c = ax.pcolor(X, Y, initial_surf, cmap=plt.cm.jet)
    # fig.colorbar(c, ax=ax)
    # plt.show()

    # [H, final_surf, dtlist] = sia(nx, ny, H, delta_t, final_time, dx, dy, X, Y, bed_scaled, M, A)

    # for k in range(0,1):
    #     print("time block %.1f" % k)
    #     [H, final_surf, dtlist] = sia(nx, ny, H, delta_t, final_time, dx, dy, X, Y, bed_scaled, M, A)
    #     # if any(any(H<0)), error('negative output thicknesses detected'), end
    #     # vol = [vol printvolume(k*tblocka,dx,dy,H)]

    fig, axes = plt.subplots(1,2, constrained_layout=True)
    fig.suptitle("Evolution of Greenland Surface")
    im1 = axes[0].pcolor(X/1000, Y/1000, initial_surf, cmap='RdBu_r')
    im2 = axes[1].pcolor(X/1000, Y/1000,initial_surf, cmap='RdBu_r')
    axes[0].set_title("initial surface")
    axes[1].set_title("final surface")
    cbar_ax = fig.colorbar(im1, ax=axes[:])
    cbar_ax.set_label('surface height (m)')
    fig.supxlabel('kilometers')
    fig.supylabel('kilometers')
    plt.show()

    
def getSurface(H, b):
    f = 910.0 / 1028.0                 # fraction of floating ice below surface
    h = H + b                          # only valid where H > 0 and grounded

    for i in range(len(H)):
        for j in range(len(H[0])):
            if H[i,j] <= 0.0:
                h[i,j] = b[i,j]
            if b[i,j] <0.0:
                h[i,j] = 0.0
            if (H[i,j] > 0.0) and (b[i,j] < -f*H[i,j]):
                h[i,j] = (1-f)*H[i,j]
    return h

if __name__=="__main__":
    greenland()
    # roughice() # model of evolution of very rough ice shelf