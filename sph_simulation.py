"""SPH (Smoothed Particle Hydrodynamics) 2D dam-break simulation using @metal_kernel.

Dam-break scenario: a tall block of water on the left collapses under gravity
and flows rightward across the [0,1]x[0,1] domain.

Two GPU kernels per timestep (hash-grid accelerated, O(N*k) instead of O(N^2)):
  1. compute_density  — poly6 kernel, iterates over 3x3 neighboring cells
  2. update_particles — pressure (spiky) + viscosity forces, symplectic Euler

SPH parameter derivation:
  dx    = 0.01       particle spacing
  h     = 0.025      smoothing length (2.5 * dx, ~20 neighbors in 2D)
  rho0  = 1000       rest density
  mass  = rho0*dx^2  = 0.1   (ensures SPH density ~ rho0 for uniform grid)
  cs    ~ 32         speed of sound (10x max expected velocity ~3.1 m/s)
  k     = 1000       stiffness  (~ cs^2, linear EOS)
  mu    = 2.0        viscosity
  dt    = 0.0001     timestep   (CFL: dt < 0.4*h/cs ~ 0.0003)
  gw    = 40         hash grid width (ceil(1.0 / h))
"""

from metal_kernel import metal_kernel
import math
import sys
sys.path.insert(0, "metal_runtime/build")
from metal_backend import MetalRenderer

# ── Hash grid constants ──────────────────────────────────────────────────────

H = 0.025          # smoothing length = cell size
GRID_W = 40        # ceil(1.0 / H)
NUM_CELLS = GRID_W * GRID_W  # 1600

# ── Kernel 1: compute density (hash-grid accelerated) ───────────────────────

@metal_kernel
def compute_density(pos_x, pos_y, density,
                    cell_start, cell_count, sorted_idx, n, tid):
    if tid < n:
        h = 0.025
        mass = 0.1
        gw = 40
        h2 = h * h
        h4 = h2 * h2
        h8 = h4 * h4
        poly6 = 4.0 / (3.14159265 * h8)

        pos_x[tid] = pos_x[tid] * 1.0
        pos_y[tid] = pos_y[tid] * 1.0

        xi = pos_x[tid]
        yi = pos_y[tid]
        rho = 0.0

        cell_xi = xi // h
        cell_yi = yi // h
        if cell_xi > gw - 1:
            cell_xi = gw - 1
        if cell_yi > gw - 1:
            cell_yi = gw - 1

        for di in range(0, 3):
            for dj in range(0, 3):
                ni = cell_yi + di - 1
                nj = cell_xi + dj - 1
                if ni >= 0 and ni < gw and nj >= 0 and nj < gw:
                    cid = ni * gw + nj
                    cs = cell_start[cid]
                    cc = cell_count[cid]
                    for k in range(cs, cs + cc):
                        j = sorted_idx[k]
                        dpx = xi - pos_x[j]
                        dpy = yi - pos_y[j]
                        r2 = dpx * dpx + dpy * dpy
                        if r2 < h2:
                            diff = h2 - r2
                            rho += mass * poly6 * diff * diff * diff

        density[tid] = rho

# ── Kernel 2: forces + integration (hash-grid accelerated) ──────────────────
#
# Standard symmetric SPH momentum equation:
#   a_i = -sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2) grad_W_ij + viscosity + gravity

@metal_kernel
def update_particles(pos_x, pos_y, vel_x, vel_y, density,
                     new_pos_x, new_pos_y, new_vel_x, new_vel_y,
                     cell_start, cell_count, sorted_idx, n, tid):
    if tid < n:
        h = 0.025
        mass = 0.1
        rho0 = 1000.0
        k_stiff = 1000.0
        mu = 2.0
        dt = 0.0001
        grav = -9.81
        eps = 0.00001
        gw = 40

        h2 = h * h
        h5 = h2 * h2 * h
        spiky_grad = -30.0 / (3.14159265 * h5)
        visc_lap = 40.0 / (3.14159265 * h5)

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

        pi_press = k_stiff * (rhoi - rho0)
        rhoi2 = rhoi * rhoi

        ax = 0.0
        ay = 0.0

        cell_xi = xi // h
        cell_yi = yi // h
        if cell_xi > gw - 1:
            cell_xi = gw - 1
        if cell_yi > gw - 1:
            cell_yi = gw - 1

        for di in range(0, 3):
            for dj in range(0, 3):
                ni = cell_yi + di - 1
                nj = cell_xi + dj - 1
                if ni >= 0 and ni < gw and nj >= 0 and nj < gw:
                    cid = ni * gw + nj
                    cs = cell_start[cid]
                    cc = cell_count[cid]
                    for k in range(cs, cs + cc):
                        j = sorted_idx[k]
                        if j == tid:
                            continue
                        dpx = xi - pos_x[j]
                        dpy = yi - pos_y[j]
                        r2 = dpx * dpx + dpy * dpy

                        if r2 < h2 and r2 > eps:
                            r = sqrt(r2) * 1.0
                            rhoj = density[j]
                            rhoj2 = rhoj * rhoj

                            pj_press = k_stiff * (rhoj - rho0)

                            hr = h - r
                            dWdr = spiky_grad * hr * hr

                            # Pressure acceleration (symmetric form)
                            press_acc = mass * (pi_press / rhoi2 + pj_press / rhoj2) * dWdr
                            ax += press_acc * dpx / r * (-1.0)
                            ay += press_acc * dpy / r * (-1.0)

                            # Viscosity acceleration
                            lap = visc_lap * (h - r)
                            visc_coeff = mu * mass / (rhoi * rhoj) * lap
                            ax += visc_coeff * (vel_x[j] - vxi)
                            ay += visc_coeff * (vel_y[j] - vyi)

        # Gravity
        ay += grav

        # Symplectic Euler integration
        nvx = vxi + dt * ax
        nvy = vyi + dt * ay
        nx = xi + dt * nvx
        ny = yi + dt * nvy

        # Boundary clamping [0, 1] with velocity reflection
        damping = 0.3
        if nx < 0.0:
            nx = eps
            if nvx < 0.0:
                nvx = nvx * (-1.0) * damping
        if nx > 1.0:
            nx = 1.0 - eps
            if nvx > 0.0:
                nvx = nvx * (-1.0) * damping
        if ny < 0.0:
            ny = eps
            if nvy < 0.0:
                nvy = nvy * (-1.0) * damping
        if ny > 1.0:
            ny = 1.0 - eps
            if nvy > 0.0:
                nvy = nvy * (-1.0) * damping

        new_pos_x[tid] = nx
        new_pos_y[tid] = ny
        new_vel_x[tid] = nvx
        new_vel_y[tid] = nvy


# ── Build hash grid (Python-side, O(N)) ─────────────────────────────────────

def build_grid(pos_x, pos_y, N):
    """Count-sort particles into grid cells.  Returns (cell_start, cell_count, sorted_idx)."""
    cell_counts = [0] * NUM_CELLS
    particle_cell = [0] * N

    for i in range(N):
        cx = max(0, min(GRID_W - 1, int(pos_x[i] / H)))
        cy = max(0, min(GRID_W - 1, int(pos_y[i] / H)))
        c = cy * GRID_W + cx
        particle_cell[i] = c
        cell_counts[c] += 1

    # Prefix sum → cell_start
    cell_starts = [0] * NUM_CELLS
    for c in range(1, NUM_CELLS):
        cell_starts[c] = cell_starts[c - 1] + cell_counts[c - 1]

    # Scatter particle indices into sorted array
    sorted_idx = [0] * N
    offsets = cell_starts[:]
    for i in range(N):
        c = particle_cell[i]
        sorted_idx[offsets[c]] = i
        offsets[c] += 1

    return cell_starts, cell_counts, sorted_idx


# ── Initial conditions: dam-break block on the left ─────────────────────────
# Particles fill [0.005, 0.295] x [0.005, 0.595] with spacing dx=0.01

dx = 0.01
cols, rows = [], []
c = dx * 0.5
while c < 0.30:
    cols.append(c)
    c += dx
r = dx * 0.5
while r < 0.60:
    rows.append(r)
    r += dx

pos_x, pos_y = [], []
for ry in rows:
    for cx in cols:
        pos_x.append(cx)
        pos_y.append(ry)

N = len(pos_x)
vel_x = [0.0] * N
vel_y = [0.0] * N

# ── Print generated Metal source ────────────────────────────────────────────

print("=== Generated Metal: compute_density ===")
print(compute_density.metal_source)
print("=== Generated Metal: update_particles ===")
print(update_particles.metal_source)

# ── Create Metal renderer ──────────────────────────────────────────────────

renderer = MetalRenderer(800, 800)

# ── Simulation loop ─────────────────────────────────────────────────────────

num_steps = 10000
print_every = 200

print(f"\nRunning SPH dam-break: {N} particles, {num_steps} steps (hash-grid accelerated)")
print(f"Initial center of mass: x={sum(pos_x)/N:.4f}, y={sum(pos_y)/N:.4f}\n")

for step in range(num_steps):
    if not renderer.poll_events():
        print("Window closed, stopping simulation.")
        break

    # Build spatial hash grid (O(N))
    cell_starts, cell_counts, sorted_idx = build_grid(pos_x, pos_y, N)

    # Step 1: Compute density
    density_result = compute_density.launch(grid_size=N, buffers=[
        {"name": "pos_x",      "type": "float", "data": pos_x},
        {"name": "pos_y",      "type": "float", "data": pos_y},
        {"name": "density",    "type": "float", "size": N},
        {"name": "cell_start", "type": "int",   "data": cell_starts},
        {"name": "cell_count", "type": "int",   "data": cell_counts},
        {"name": "sorted_idx", "type": "int",   "data": sorted_idx},
        {"name": "n",          "type": "uint",  "value": N},
    ])
    density = list(density_result["density"])

    # Step 2: Compute forces and integrate
    update_result = update_particles.launch(grid_size=N, buffers=[
        {"name": "pos_x",      "type": "float", "data": pos_x},
        {"name": "pos_y",      "type": "float", "data": pos_y},
        {"name": "vel_x",      "type": "float", "data": vel_x},
        {"name": "vel_y",      "type": "float", "data": vel_y},
        {"name": "density",    "type": "float", "data": density},
        {"name": "new_pos_x",  "type": "float", "size": N},
        {"name": "new_pos_y",  "type": "float", "size": N},
        {"name": "new_vel_x",  "type": "float", "size": N},
        {"name": "new_vel_y",  "type": "float", "size": N},
        {"name": "cell_start", "type": "int",   "data": cell_starts},
        {"name": "cell_count", "type": "int",   "data": cell_counts},
        {"name": "sorted_idx", "type": "int",   "data": sorted_idx},
        {"name": "n",          "type": "uint",  "value": N},
    ])

    # Step 3: Swap buffers
    pos_x = list(update_result["new_pos_x"])
    pos_y = list(update_result["new_pos_y"])
    vel_x = list(update_result["new_vel_x"])
    vel_y = list(update_result["new_vel_y"])

    # Step 4: Render frame
    renderer.render_frame(pos_x, pos_y, vel_x, vel_y)

    # Step 5: Print summary
    if (step + 1) % print_every == 0:
        cx = sum(pos_x) / N
        cy = sum(pos_y) / N
        avg_rho = sum(density) / N
        max_v = max(math.sqrt(vx * vx + vy * vy) for vx, vy in zip(vel_x, vel_y))
        print(f"Step {step+1:5d}: center=({cx:.4f}, {cy:.4f})  "
              f"avg_density={avg_rho:.1f}  max_vel={max_v:.4f}")

# ── Final ASCII visualization ─────────────────────────────────────────────

print("\n=== ASCII visualization (domain [0,1] x [0,1]) ===")
grid_w, grid_h = 60, 30
grid = [['.' for _ in range(grid_w)] for _ in range(grid_h)]

for i in range(N):
    gx = int(pos_x[i] * (grid_w - 1))
    gy = int(pos_y[i] * (grid_h - 1))
    gx = max(0, min(grid_w - 1, gx))
    gy = max(0, min(grid_h - 1, gy))
    grid[grid_h - 1 - gy][gx] = '#'

print('+' + '-' * grid_w + '+')
for row in grid:
    print('|' + ''.join(row) + '|')
print('+' + '-' * grid_w + '+')
print("  '#' = particle position")
