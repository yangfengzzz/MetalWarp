"""SPH (Smoothed Particle Hydrodynamics) 2D dam-break simulation using @metal_kernel.

Dam-break scenario: a tall block of water on the left collapses under gravity
and flows rightward across the [0,1]x[0,1] domain.

All simulation state and hash-grid buffers are allocated on GPU. Grid construction
is done by Metal compute kernels (no Python-side build_grid).
"""

import math
from pathlib import Path
import sys

from .metal_kernel import metal_kernel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "metal_runtime" / "build"))
from metal_backend import MetalRenderer, MetalDevice

# ── Hash grid constants ──────────────────────────────────────────────────────

H = 0.025
GRID_W = 40
NUM_CELLS = GRID_W * GRID_W

@metal_kernel
def count_particles_per_cell(pos_x, pos_y, cell_count, n, num_cells, tid):
    if tid < n:
        pos_x[tid] = pos_x[tid] * 1.0
        pos_y[tid] = pos_y[tid] * 1.0

    if tid < num_cells:
        h = 0.025
        gw = 40
        count = 0
        cid = tid

        cell_y = cid // gw
        cell_x = cid - cell_y * gw

        for i in range(0, n):
            px = pos_x[i] * 1.0
            py = pos_y[i] * 1.0
            cx = px // h
            cy = py // h
            if cx < 0:
                cx = 0
            if cx > gw - 1:
                cx = gw - 1
            if cy < 0:
                cy = 0
            if cy > gw - 1:
                cy = gw - 1
            if cx == cell_x and cy == cell_y:
                count += 1

        cell_count[cid] = count


@metal_kernel
def prefix_sum_cell_counts(cell_count, cell_start, num_cells, tid):
    if tid == 0 and num_cells > 0:
        cell_start[0] = 0
        for c in range(1, num_cells):
            cell_start[c] = cell_start[c - 1] + cell_count[c - 1]


@metal_kernel
def scatter_particles_by_cell(pos_x, pos_y, cell_start, sorted_idx, n, num_cells, tid):
    if tid < n:
        pos_x[tid] = pos_x[tid] * 1.0
        pos_y[tid] = pos_y[tid] * 1.0

    if tid < num_cells:
        h = 0.025
        gw = 40
        cid = tid
        write_ptr = cell_start[cid]

        cell_y = cid // gw
        cell_x = cid - cell_y * gw

        for i in range(0, n):
            px = pos_x[i] * 1.0
            py = pos_y[i] * 1.0
            cx = px // h
            cy = py // h
            if cx < 0:
                cx = 0
            if cx > gw - 1:
                cx = gw - 1
            if cy < 0:
                cy = 0
            if cy > gw - 1:
                cy = gw - 1
            if cx == cell_x and cy == cell_y:
                sorted_idx[write_ptr] = i
                write_ptr += 1


# ── Kernel 1: compute density (hash-grid accelerated) ───────────────────────

@metal_kernel
def compute_density(pos_x, pos_y, density,
                    cell_start, cell_count, sorted_idx, mass: float, n, tid):
    if tid < n:
        h = 0.025
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

@metal_kernel
def update_particles(pos_x, pos_y, vel_x, vel_y, density,
                     new_pos_x, new_pos_y, new_vel_x, new_vel_y,
                     cell_start, cell_count, sorted_idx, mass: float, n, tid):
    if tid < n:
        h = 0.025
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

                            press_acc = mass * (pi_press / rhoi2 + pj_press / rhoj2) * dWdr
                            ax += press_acc * dpx / r * (-1.0)
                            ay += press_acc * dpy / r * (-1.0)

                            lap = visc_lap * (h - r)
                            visc_coeff = mu * mass / (rhoi * rhoj) * lap
                            ax += visc_coeff * (vel_x[j] - vxi)
                            ay += visc_coeff * (vel_y[j] - vyi)

        ay += grav

        nvx = vxi + dt * ax
        nvy = vyi + dt * ay
        nx = xi + dt * nvx
        ny = yi + dt * nvy

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


# ── Initial conditions: dam-break block on the left ─────────────────────────


dx = 0.006
rho0 = 1000.0
particle_mass = rho0 * dx * dx
cols, rows = [], []
c = dx * 0.5
while c < 0.30:
    cols.append(c)
    c += dx
r = dx * 0.5
while r < 0.60:
    rows.append(r)
    r += dx

init_pos_x, init_pos_y = [], []
for ry in rows:
    for cx in cols:
        init_pos_x.append(cx)
        init_pos_y.append(ry)

N = len(init_pos_x)
init_vel_x = [0.0] * N
init_vel_y = [0.0] * N

print("=== Generated Metal: compute_density ===")
print(compute_density.metal_source)
print("=== Generated Metal: update_particles ===")
print(update_particles.metal_source)

device = MetalDevice()
renderer = MetalRenderer(device, 800, 800)

# ── Persistent GPU buffers (all simulation state lives on GPU) ─────────────

pos_x_buf = device.create_buffer_with_data("float", init_pos_x)
pos_y_buf = device.create_buffer_with_data("float", init_pos_y)
vel_x_buf = device.create_buffer_with_data("float", init_vel_x)
vel_y_buf = device.create_buffer_with_data("float", init_vel_y)

density_buf = device.create_buffer("float", N)
new_pos_x_buf = device.create_buffer("float", N)
new_pos_y_buf = device.create_buffer("float", N)
new_vel_x_buf = device.create_buffer("float", N)
new_vel_y_buf = device.create_buffer("float", N)

cell_start_buf = device.create_buffer("int", NUM_CELLS)
cell_count_buf = device.create_buffer("int", NUM_CELLS)
sorted_idx_buf = device.create_buffer("int", N)

n_buf = device.create_scalar_buffer("uint", N)
num_cells_buf = device.create_scalar_buffer("uint", NUM_CELLS)
num_cells_int_buf = device.create_scalar_buffer("int", NUM_CELLS)
mass_buf = device.create_scalar_buffer("float", particle_mass)

num_steps = 10000
print_every = 200

print(f"\nRunning SPH dam-break: {N} particles, {num_steps} steps (GPU hash-grid)")
print(f"Initial center of mass: x={sum(init_pos_x)/N:.4f}, y={sum(init_pos_y)/N:.4f}\n")

for step in range(num_steps):
    if not renderer.poll_events():
        print("Window closed, stopping simulation.")
        break

    # Step 0: Build spatial hash grid on GPU
    device.run_kernel_with_buffers(
        count_particles_per_cell.metal_source,
        "count_particles_per_cell",
        NUM_CELLS,
        [pos_x_buf, pos_y_buf, cell_count_buf, n_buf, num_cells_buf],
    )
    device.run_kernel_with_buffers(
        prefix_sum_cell_counts.metal_source,
        "prefix_sum_cell_counts",
        1,
        [cell_count_buf, cell_start_buf, num_cells_int_buf],
    )
    device.run_kernel_with_buffers(
        scatter_particles_by_cell.metal_source,
        "scatter_particles_by_cell",
        NUM_CELLS,
        [pos_x_buf, pos_y_buf, cell_start_buf, sorted_idx_buf, n_buf, num_cells_buf],
    )

    # Step 1: Compute density
    device.run_kernel_with_buffers(
        compute_density.metal_source,
        "compute_density",
        N,
        [pos_x_buf, pos_y_buf, density_buf, cell_start_buf, cell_count_buf, sorted_idx_buf, mass_buf, n_buf],
    )

    # Step 2: Compute forces + integrate into new buffers
    device.run_kernel_with_buffers(
        update_particles.metal_source,
        "update_particles",
        N,
        [
            pos_x_buf,
            pos_y_buf,
            vel_x_buf,
            vel_y_buf,
            density_buf,
            new_pos_x_buf,
            new_pos_y_buf,
            new_vel_x_buf,
            new_vel_y_buf,
            cell_start_buf,
            cell_count_buf,
            sorted_idx_buf,
            mass_buf,
            n_buf,
        ],
    )

    # Step 3: Swap state buffers (GPU handle swap, no CPU copy)
    pos_x_buf, new_pos_x_buf = new_pos_x_buf, pos_x_buf
    pos_y_buf, new_pos_y_buf = new_pos_y_buf, pos_y_buf
    vel_x_buf, new_vel_x_buf = new_vel_x_buf, vel_x_buf
    vel_y_buf, new_vel_y_buf = new_vel_y_buf, vel_y_buf

    # Step 4: Render directly from persistent GPU simulation buffers
    renderer.render_frame_from_buffers(device, pos_x_buf, pos_y_buf, vel_x_buf, vel_y_buf)

    if (step + 1) % print_every == 0:
        pos_x = device.download_buffer(pos_x_buf)
        pos_y = device.download_buffer(pos_y_buf)
        vel_x = device.download_buffer(vel_x_buf)
        vel_y = device.download_buffer(vel_y_buf)
        density = device.download_buffer(density_buf)
        cx = sum(pos_x) / N
        cy = sum(pos_y) / N
        avg_rho = sum(density) / N
        max_v = max(math.sqrt(vx * vx + vy * vy) for vx, vy in zip(vel_x, vel_y))
        print(f"Step {step+1:5d}: center=({cx:.4f}, {cy:.4f})  "
              f"avg_density={avg_rho:.1f}  max_vel={max_v:.4f}")

pos_x = device.download_buffer(pos_x_buf)
pos_y = device.download_buffer(pos_y_buf)

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
