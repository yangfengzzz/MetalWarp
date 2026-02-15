"""SPH (Smoothed Particle Hydrodynamics) 2D fluid simulation using @metal_kernel.

Dam-break scenario: a block of particles on the left collapses under gravity.
Two GPU kernels per timestep:
  1. compute_density  — poly6 kernel, O(N²) brute force
  2. update_particles — pressure (spiky) + viscosity forces, Euler integration
"""

from metal_kernel import metal_kernel
import math

# ── Kernel 1: compute density for each particle ─────────────────────────────

@metal_kernel
def compute_density(pos_x, pos_y, density, n, tid):
    if tid < n:
        # SPH parameters (baked as local constants)
        h = 0.04
        mass = 0.02
        # Poly6 coefficient for 2D: 4 / (pi * h^8)
        h2 = h * h
        h4 = h2 * h2
        h8 = h4 * h4
        poly6 = 4.0 / (3.14159265 * h8)

        # Force float type for input arrays (identity write)
        pos_x[tid] = pos_x[tid] * 1.0
        pos_y[tid] = pos_y[tid] * 1.0

        xi = pos_x[tid]
        yi = pos_y[tid]
        rho = 0.0

        for j in range(n):
            dx = xi - pos_x[j]
            dy = yi - pos_y[j]
            r2 = dx * dx + dy * dy
            if r2 < h2:
                diff = h2 - r2
                rho += mass * poly6 * diff * diff * diff

        density[tid] = rho

# ── Kernel 2: compute forces, integrate, and apply boundary conditions ──────

@metal_kernel
def update_particles(pos_x, pos_y, vel_x, vel_y, density,
                     new_pos_x, new_pos_y, new_vel_x, new_vel_y, n, tid):
    if tid < n:
        # SPH parameters
        h = 0.04
        mass = 0.02
        rho0 = 1000.0
        k_stiff = 200.0
        mu = 50.0
        dt = 0.00005
        grav = -9.81
        eps = 0.00001

        # Kernel coefficients (2D)
        # Spiky gradient: -30 / (pi * h^5)
        h2 = h * h
        h5 = h2 * h2 * h
        spiky_grad = -30.0 / (3.14159265 * h5)
        # Viscosity Laplacian: 40 / (pi * h^5)
        visc_lap = 40.0 / (3.14159265 * h5)

        # Force float type for input arrays (identity write)
        pos_x[tid] = pos_x[tid] * 1.0
        pos_y[tid] = pos_y[tid] * 1.0
        vel_x[tid] = vel_x[tid] * 1.0
        vel_y[tid] = vel_y[tid] * 1.0
        density[tid] = density[tid] * 1.0

        xi = pos_x[tid]
        yi = pos_y[tid]
        vxi = vel_x[tid]
        vyi = vel_y[tid]
        rhoi = density[tid]

        # Pressure from equation of state
        pi_press = k_stiff * (rhoi - rho0)

        fx = 0.0
        fy = 0.0

        for j in range(n):
            if j == tid:
                continue
            dx = xi - pos_x[j]
            dy = yi - pos_y[j]
            r2 = dx * dx + dy * dy

            if r2 < h2 and r2 > eps:
                # r = sqrt(r2); use * 1.0 to force float inference
                r = sqrt(r2) * 1.0
                rhoj = density[j]

                # Pressure of neighbor
                pj_press = k_stiff * (rhoj - rho0)

                # Spiky gradient magnitude: spiky_grad * (h - r)^2 / r
                hr = h - r
                grad_mag = spiky_grad * hr * hr / r

                # Pressure force: -m * (pi + pj) / (2 * rhoj) * grad * (dx/r, dy/r)
                press_term = mass * (pi_press + pj_press) / (2.0 * rhoj)
                fx += press_term * grad_mag * dx / r * (-1.0)
                fy += press_term * grad_mag * dy / r * (-1.0)

                # Viscosity force: mu * m * (vj - vi) / rhoj * laplacian
                lap = visc_lap * (h - r)
                fx += mu * mass * (vel_x[j] - vxi) / rhoj * lap
                fy += mu * mass * (vel_y[j] - vyi) / rhoj * lap

        # Add gravity
        fy += grav

        # Integrate (Euler)
        inv_rho = 1.0 / (rhoi + eps)
        nvx = vxi + dt * fx * inv_rho
        nvy = vyi + dt * fy * inv_rho
        nx = xi + dt * nvx
        ny = yi + dt * nvy

        # Boundary clamping [0, 1] x [0, 1] with reflection
        damping = 0.5
        if nx < 0.0:
            nx = 0.0
            nvx = nvx * (-1.0) * damping
        if nx > 1.0:
            nx = 1.0
            nvx = nvx * (-1.0) * damping
        if ny < 0.0:
            ny = 0.0
            nvy = nvy * (-1.0) * damping
        if ny > 1.0:
            ny = 1.0
            nvy = nvy * (-1.0) * damping

        new_pos_x[tid] = nx
        new_pos_y[tid] = ny
        new_vel_x[tid] = nvx
        new_vel_y[tid] = nvy


# ── Initial conditions: 10x10 grid in [0.1, 0.3] x [0.1, 0.3] ─────────────

N = 100
spacing = 0.015
pos_x = []
pos_y = []
for row in range(10):
    for col in range(10):
        pos_x.append(0.1 + col * spacing)
        pos_y.append(0.1 + row * spacing)

vel_x = [0.0] * N
vel_y = [0.0] * N

# ── Print generated Metal source ────────────────────────────────────────────

print("=== Generated Metal: compute_density ===")
print(compute_density.metal_source)
print("=== Generated Metal: update_particles ===")
print(update_particles.metal_source)

# ── Simulation loop ─────────────────────────────────────────────────────────

num_steps = 200
print_every = 50

print(f"\nRunning SPH simulation: {N} particles, {num_steps} steps")
print(f"Initial center of mass: x={sum(pos_x)/N:.4f}, y={sum(pos_y)/N:.4f}\n")

for step in range(num_steps):
    # Step 1: Compute density
    density_buffers = [
        {"name": "pos_x",   "type": "float", "data": pos_x},
        {"name": "pos_y",   "type": "float", "data": pos_y},
        {"name": "density", "type": "float", "size": N},
        {"name": "n",       "type": "uint",  "value": N},
    ]
    density_result = compute_density.launch(grid_size=N, buffers=density_buffers)
    density = list(density_result["density"])

    # Step 2: Compute forces and integrate
    update_buffers = [
        {"name": "pos_x",     "type": "float", "data": pos_x},
        {"name": "pos_y",     "type": "float", "data": pos_y},
        {"name": "vel_x",     "type": "float", "data": vel_x},
        {"name": "vel_y",     "type": "float", "data": vel_y},
        {"name": "density",   "type": "float", "data": density},
        {"name": "new_pos_x", "type": "float", "size": N},
        {"name": "new_pos_y", "type": "float", "size": N},
        {"name": "new_vel_x", "type": "float", "size": N},
        {"name": "new_vel_y", "type": "float", "size": N},
        {"name": "n",         "type": "uint",  "value": N},
    ]
    update_result = update_particles.launch(grid_size=N, buffers=update_buffers)

    # Step 3: Swap buffers
    pos_x = list(update_result["new_pos_x"])
    pos_y = list(update_result["new_pos_y"])
    vel_x = list(update_result["new_vel_x"])
    vel_y = list(update_result["new_vel_y"])

    # Step 4: Print summary
    if (step + 1) % print_every == 0:
        cx = sum(pos_x) / N
        cy = sum(pos_y) / N
        avg_rho = sum(density) / N
        max_v = max(math.sqrt(vx * vx + vy * vy) for vx, vy in zip(vel_x, vel_y))
        print(f"Step {step+1:4d}: center=({cx:.4f}, {cy:.4f})  "
              f"avg_density={avg_rho:.4f}  max_vel={max_v:.4f}")

# ── Final output ────────────────────────────────────────────────────────────

print("\n=== Final particle positions ===")
for i in range(N):
    print(f"  P{i:3d}: ({pos_x[i]:.4f}, {pos_y[i]:.4f})  "
          f"vel=({vel_x[i]:.4f}, {vel_y[i]:.4f})")

# ── ASCII visualization ────────────────────────────────────────────────────

print("\n=== ASCII visualization (domain [0,1] x [0,1]) ===")
grid_w, grid_h = 60, 30
grid = [['.' for _ in range(grid_w)] for _ in range(grid_h)]

for i in range(N):
    gx = int(pos_x[i] * (grid_w - 1))
    gy = int(pos_y[i] * (grid_h - 1))
    gx = max(0, min(grid_w - 1, gx))
    gy = max(0, min(grid_h - 1, gy))
    # Flip y so bottom of domain is bottom of display
    grid[grid_h - 1 - gy][gx] = '#'

print('+' + '-' * grid_w + '+')
for row in grid:
    print('|' + ''.join(row) + '|')
print('+' + '-' * grid_w + '+')
print("  '#' = particle position")
