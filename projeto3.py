import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def kerr_geodesics(tau, y, a, M, E, L, Q):
    """
    Defines the first-order geodesic equations for a Kerr Spacetime.
    y = [t, r, theta, phi]
    """
    t, r, theta, phi = y
    
    # Helpful auxiliary variables
    sigma = r**2 + a**2 * np.cos(theta)**2
    delta = r**2 - 2*M*r + a**2
    
    # Potential terms for r and theta
    # P is related to energy and angular momentum
    P = E * (r**2 + a**2) - a * L
    
    # Radial equation: (sigma * dr/dtau)^2 = R_func
    R_func = P**2 - delta * (r**2 + (L - a*E)**2 + Q)
    
    # Angular equation: (sigma * dtheta/dtau)^2 = Theta_func
    Theta_func = Q - (np.cos(theta)**2 / np.sin(theta)**2) * L**2 + a**2 * (E**2 - 1) * np.cos(theta)**2
    
    # Ensure potentials aren't negative due to numerical precision
    R_func = max(0, R_func)
    Theta_func = max(0, Theta_func)
    
    # First-order ODEs (dr/dtau and dtheta/dtau signs depend on initial direction)
    dt_dtau = ( (r**2 + a**2) * P / (sigma * delta) ) - (a * (a * E * np.sin(theta)**2 - L) / (sigma))
    dr_dtau = -np.sqrt(R_func) / sigma  # Negative for an inspiralling/plunging particle
    dtheta_dtau = np.sqrt(Theta_func) / sigma
    dphi_dtau = ( (a * P) / (sigma * delta) ) - ( (a * E - L / np.sin(theta)**2) / sigma )
    
    return [dt_dtau, dr_dtau, dtheta_dtau, dphi_dtau]

# --- 1. Parameters & Constants ---
M = 1.0          # Mass of the Black Hole
a = 0.9          # Spin parameter (0 <= a < M)
mu = 1.0         # Mass of the particle (test mass)

# Constants of Motion (Choosing values for a plunging/spiraling orbit)
E = 0.94         # Energy per unit mass
L = 2.0          # Angular momentum per unit mass
Q = 2.0          # Carter constant (non-zero for non-equatorial orbits)

# --- 2. Initial Conditions ---
# [t, r, theta, phi]
y0 = [0.0, 10.0, np.pi/3, 0.0] 
tau_span = (0, 800) # Proper time range to integrate
tau_eval = np.linspace(tau_span[0], tau_span[1], 5000)

# --- 3. Integration ---
# Using DOP853 for high-precision as requested
sol = solve_ivp(
    kerr_geodesics, 
    tau_span, 
    y0, 
    args=(a, M, E, L, Q), 
    method='DOP853', 
    t_eval=tau_eval,
    rtol=1e-11, 
    atol=1e-13
)

# --- 4. Post-Processing: Conversion to Quasi-Cartesian ---
r = sol.y[1]
theta = sol.y[2]
phi = sol.y[3]

x = np.sqrt(r**2 + a**2) * np.sin(theta) * np.cos(phi)
y = np.sqrt(r**2 + a**2) * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# --- 5. Visualization ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Particle Trajectory')
ax.scatter(0, 0, 0, color='black', s=100, label='Black Hole')
ax.set_title(f"Geodesic around a Kerr BH (a={a})")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------------


# Assuming 'sol' is the output from the previous DOP853 integration
# sol.y[0] is coordinate time 't'
# sol.y[1, 2, 3] are r, theta, phi

t_coords = sol.y[0]
r_coords = sol.y[1]
theta_coords = sol.y[2]
phi_coords = sol.y[3]

# 1. Convert the raw Boyer-Lindquist arrays to Quasi-Cartesian
x_raw = np.sqrt(r_coords**2 + a**2) * np.sin(theta_coords) * np.cos(phi_coords)
y_raw = np.sqrt(r_coords**2 + a**2) * np.sin(theta_coords) * np.sin(phi_coords)
z_raw = r_coords * np.cos(theta_coords)

# 2. We need a uniform grid of 't' for the Quadrupole Formula (dt must be constant)
# We define a new time array from the start to the end of the coordinate time
t_uniform = np.linspace(t_coords[0], t_coords[-1], len(t_coords))

# 3. Interpolate x, y, z from 'proper time' mapping to 'uniform coordinate time'
# This is the crucial step that makes 't' the independent parameter
f_x = interp1d(t_coords, x_raw, kind='cubic')
f_y = interp1d(t_coords, y_raw, kind='cubic')
f_z = interp1d(t_coords, z_raw, kind='cubic')

x_t = f_x(t_uniform)
y_t = f_y(t_uniform)
z_t = f_z(t_uniform)

# Now you have x_t, y_t, z_t defined on a uniform clock (t_uniform).
# You can now calculate the second derivative d^2/dt^2 safely!

# Example: The first component of the Quadrupole Tensor I_xx = mass * x^2
I_xx = mu * x_t**2

# To get the second derivative:
dt = t_uniform[1] - t_uniform[0]
d2I_xx_dt2 = np.gradient(np.gradient(I_xx, dt), dt)

#-----------------------------------------------------------------------------------------------

# --- 1. Setup the Tensor Arrays ---
# x_t, y_t, z_t are our interpolated arrays from the previous step
# t_uniform is our uniform time grid with spacing 'dt'
num_points = len(t_uniform)
dt = t_uniform[1] - t_uniform[0]

# We will store the 3x3 reduced quadrupole moment for each time step
# Shape: (3, 3, num_points)
Q = np.zeros((3, 3, num_points))

# --- 2. Calculate the Reduced Quadrupole Moment ---
# r_sq is the squared distance from the origin (the Black Hole)
r_sq = x_t**2 + y_t**2 + z_t**2

# We populate the matrix components manually for clarity
Q[0, 0] = mu * (x_t * x_t - (1/3) * r_sq)  # Qxx
Q[1, 1] = mu * (y_t * y_t - (1/3) * r_sq)  # Qyy
Q[2, 2] = mu * (z_t * z_t - (1/3) * r_sq)  # Qzz
Q[0, 1] = Q[1, 0] = mu * (x_t * y_t)       # Qxy
Q[0, 2] = Q[2, 0] = mu * (x_t * z_t)       # Qxz
Q[1, 2] = Q[2, 1] = mu * (y_t * z_t)       # Qyz

# --- 3. Compute the Second Time Derivatives (Q-double-dot) ---
# We need d^2/dt^2 for the gravitational wave formula.
# We apply np.gradient twice to every component of the 3x3 matrix.
Q_ddot = np.zeros((3, 3, num_points))

for i in range(3):
    for j in range(3):
        # First derivative (velocity of the moment)
        first_deriv = np.gradient(Q[i, j], dt)
        # Second derivative (acceleration of the moment)
        Q_ddot[i, j] = np.gradient(first_deriv, dt)

# Now Q_ddot[i, j] contains the source term for the GW signal!

#-----------------------------------------------------------------------------------------------


def get_waveforms(Q_ddot, D, inclination):
    """
    Transforms the 3x3 quadrupole acceleration tensor into 
    the two observable polarizations: h_plus and h_cross.
    """
    # Physical constants (in Geometrized Units G=c=1, these are 1)
    # If you want real units, use G/c**4
    coupling = 1.0 / D 
    
    iota = inclination
    cos_i = np.cos(iota)
    sin_i = np.sin(iota)

    # Extract the components from our 3x3 Q_ddot matrix
    # Q_ddot shape is (3, 3, num_points)
    Qxx = Q_ddot[0, 0]
    Qyy = Q_ddot[1, 1]
    Qzz = Q_ddot[2, 2]
    Qxy = Q_ddot[0, 1]
    Qxz = Q_ddot[0, 2]
    Qyz = Q_ddot[1, 2]

    # --- 1. The 'Plus' Polarization (h_plus) ---
    # This formula projects the tensor onto the plane of the sky
    h_plus = (coupling) * (
        Qxx * (cos_i**2) + 
        Qyy - 
        Qzz * (sin_i**2) - 
        2 * Qxz * sin_i * cos_i
    )

    # --- 2. The 'Cross' Polarization (h_cross) ---
    # This represents the 45-degree rotated stretching
    h_cross = (coupling) * (
        2 * Qxy * cos_i - 
        2 * Qyz * sin_i
    )

    return h_plus, h_cross

# --- Execution ---
D = 10**6             # Observer distance (in units of M)
iota = np.pi / 4      # 45-degree inclination

h_p, h_c = get_waveforms(Q_ddot, D, iota)

# --- Visualization ---
plt.figure(figsize=(12, 5))
plt.plot(t_uniform, h_p, label=r'$h_+$ (Plus)')
plt.plot(t_uniform, h_c, label=r'$h_\times$ (Cross)', linestyle='--')
plt.xlabel('Coordinate Time (t)')
plt.ylabel('Strain (h)')
plt.title(f'Gravitational Wave Signal (Inclination: {iota:.2f} rad)')
plt.legend()
plt.grid(True)
plt.show()
