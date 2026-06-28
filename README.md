# KerrWave: Simulating Gravitational Waves from a Black Hole Inspiral

This project simulates a small particle orbiting and eventually inspiralling into a rotating black hole (Kerr black hole), then calculates the approximate gravitational waves produced by that motion.

The idea is to connect general relativity, numerical methods, and gravitational-wave physics into one pipeline.

## What it does

1. **Integrates Kerr geodesics**
   
   The particle's trajectory is calculated by solving the geodesic equations in Kerr spacetime.

   The motion is described using:
   - Energy (E)
   - Angular momentum (Lz)
   - Carter constant (Q)

   The code numerically solves for:

   - t(t)
   - r(t)
   - θ(t)
   - φ(t)

   using an ODE solver such as `scipy.integrate.solve_ivp`.

2. **Converts the orbit into Cartesian coordinates**

   The relativistic trajectory is converted from Boyer-Lindquist coordinates into:

   x, y, z

   using:

   x = r sin(θ) cos(φ)

   y = r sin(θ) sin(φ)

   z = r cos(θ)

   This allows the orbit to be visualised.

3. **Calculates the quadrupole moment**

   The mass distribution of the orbiting particle is approximated using the quadrupole tensor:

   I_ij = μ x_i x_j

   The second time derivative of this tensor is calculated numerically.

4. **Generates gravitational waves**

   Using the quadrupole approximation:

   h_ij = (2G/c^4D) * I''_ij

   the gravitational wave signal is extracted.

   The final output is the two gravitational wave polarizations:

   - h+(t)
   - hx(t)

   which represent the two possible distortions measured by a distant observer.

## Project structure

(Will be updated)
