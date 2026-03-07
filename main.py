"""
Asteroid Belt Visualization — Edge-on view using Pygame

Reads osculating orbital elements from astorb.dat.gz (Lowell Observatory),
computes 3D heliocentric positions, and renders an edge-on view where:
  • Screen X = distance from Sun (semi-major axis direction)
  • Screen Y = height above/below ecliptic (inclination effect)

Asteroids activate progressively based on their Mean Anomaly (M₀):
asteroid with M₀=k° starts orbiting at simulation time t=k seconds.
"""

import gzip
import math
import sys
import time as pytime

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GM_SUN = 4 * math.pi ** 2          # AU³/yr²  (with period in years)
DEG2RAD = math.pi / 180.0
TWO_PI = 2.0 * math.pi

WIDTH, HEIGHT = 1280, 800
FPS_CAP = 60
BG_COLOR = (4, 4, 12)

MAX_ASTEROIDS = 100_000

# Camera defaults (AU range visible)
DEFAULT_X_CENTER = 3.0              # AU — center of view
DEFAULT_X_SPAN = 8.0                # AU — total horizontal span
# cam_y_span is derived each frame as cam_x_span * HEIGHT / WIDTH (uniform scale)

# Time: simulation time in "seconds" where each second = 1° of M₀ activation
TIME_SPEED_DEFAULT = 10.0           # degrees / real-second
TIME_SPEED_MIN = 0.5
TIME_SPEED_MAX = 500.0

# Colors
SUN_COLOR = (255, 210, 80)
HUD_COLOR = (180, 200, 220)
ACTIVE_LABEL_COLOR = (100, 255, 140)

# Asteroid family identifiers
FAMILY_MAIN_BELT = 0
FAMILY_HILDA     = 1   # 3:2 MMR with Jupiter, a ≈ 3.7–4.2 AU
FAMILY_TROJAN    = 2   # Jupiter Trojans,       a ≈ 4.8–5.5 AU
FAMILY_NEO       = 3   # Near-Earth Objects,    q = a(1-e) ≤ 1.3 AU

# Planet orbits (name, a AU, ecc, inc°, ω°, Ω°, color)
ORBIT_SEGMENTS = 360
PLANETS = [
    ("Earth",    1.000, 0.0167,  0.000, 102.9,   0.0, (100, 180, 255)),
    ("Mars",     1.524, 0.0934,  1.850, 286.5,  49.6, (220, 100,  60)),
    ("Jupiter",  5.203, 0.0489,  1.304, 273.9, 100.5, (210, 170, 110)),
]

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _load_astorb_angles(filepath: str):
    """
    Read astorb.dat(.gz) and return a dict mapping asteroid number string
    to (M0_deg, omega_deg, Omega_deg, name).
    astorb.dat fixed-width columns (0-indexed):
      number : 0-5
      name   : 7-25
      M      : 115-125
      omega  : 126-136
      Omega  : 137-147
    """
    angles = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    print(f"Reading osculating angles from {filepath}...")
    t0 = pytime.time()
    with opener(filepath, 'rt', encoding='ascii', errors='replace') as f:
        for line in f:
            if len(line) < 147:
                continue
            num_str = line[0:6].strip()
            if not num_str:
                continue
            try:
                M_deg  = float(line[115:125].strip())
                om_deg = float(line[126:136].strip())
                Om_deg = float(line[137:147].strip())
                name   = line[7:25].strip()
                angles[num_str] = (M_deg, om_deg, Om_deg, name)
            except ValueError:
                continue
    print(f"  {len(angles):,} records in {pytime.time()-t0:.1f}s")
    return angles


def load_table2_asteroids(table2_path: str, astorb_path: str,
                          max_count: int = MAX_ASTEROIDS):
    """
    Load proper orbital elements from table2.dat.gz (Nesvorny+ 2024) and join
    osculating M0, omega, Omega from astorb.dat by asteroid number.

    table2.dat whitespace-separated fields (12 per line):
      0: ap(AU)  1: e_ap  2: ep  3: e_ep  4: sinIp  5: e_sinIp
      6: g  7: s  8: HMag  9: Nop  10: MPC(packed)  11: uMPC(unpacked)

    Returns dict of NumPy arrays with angles in radians, M0 in degrees.
    """
    angles_map = _load_astorb_angles(astorb_path)

    M0_list    = []
    omega_list = []
    Omega_list = []
    inc_list   = []
    ecc_list   = []
    a_list     = []
    names      = []

    opener = gzip.open if table2_path.endswith('.gz') else open
    print(f"Loading proper elements from {table2_path}...")
    t0 = pytime.time()
    count = 0
    with opener(table2_path, 'rt', encoding='ascii', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 12:
                continue
            try:
                a    = float(parts[0])
                ecc  = float(parts[2])
                sinI = float(parts[4])
                key  = parts[11].strip()
            except ValueError:
                continue

            if a <= 0 or a > 100 or ecc < 0 or ecc >= 1.0:
                continue
            if abs(sinI) > 1.0:
                continue
            if key not in angles_map:
                continue

            M0_deg, om_deg, Om_deg, name = angles_map[key]
            inc = math.asin(min(max(sinI, -1.0), 1.0))

            M0_list.append(M0_deg)
            omega_list.append(om_deg * DEG2RAD)
            Omega_list.append(Om_deg * DEG2RAD)
            inc_list.append(inc)
            ecc_list.append(ecc)
            a_list.append(a)
            names.append(name)

            count += 1
            if count >= max_count:
                break

    dt = pytime.time() - t0
    print(f"Matched {len(a_list):,} asteroids in {dt:.1f}s")

    return {
        'M0':    np.array(M0_list,    dtype=np.float64),  # degrees (trigger time)
        'omega': np.array(omega_list, dtype=np.float64),  # radians
        'Omega': np.array(Omega_list, dtype=np.float64),  # radians
        'inc':   np.array(inc_list,   dtype=np.float64),  # radians
        'ecc':   np.array(ecc_list,   dtype=np.float64),
        'a':     np.array(a_list,     dtype=np.float64),  # AU
        'names': names,
    }


def load_families_from_astorb(filepath: str):
    """
    Read all osculating elements from astorb.dat(.gz) and return Hilda,
    Jupiter Trojan, and Near-Earth Asteroid populations:
      Hildas  (3:2 MMR):  3.70 ≤ a ≤ 4.20 AU
      Trojans (L4 + L5):  4.80 ≤ a ≤ 5.50 AU
      NEOs:               q = a(1-e) ≤ 1.3 AU  (Atiras/Atens/Apollos/Amors)

    astorb.dat fixed-width columns (0-indexed):
      number : [0:6]    name  : [7:25]
      M      : [115:125]  omega : [126:136]  Omega : [137:147]
      inc    : [148:158]  ecc   : [158:168]  a     : [168:181]
    """
    opener = gzip.open if filepath.endswith('.gz') else open
    print(f"Loading Hilda/Trojan/NEO elements from {filepath}...")
    t0 = pytime.time()

    M0_h, om_h, Om_h, inc_h, ecc_h, a_h, names_h = [], [], [], [], [], [], []
    M0_t, om_t, Om_t, inc_t, ecc_t, a_t, names_t = [], [], [], [], [], [], []
    M0_n, om_n, Om_n, inc_n, ecc_n, a_n, names_n = [], [], [], [], [], [], []

    with opener(filepath, 'rt', encoding='ascii', errors='replace') as f:
        for line in f:
            if len(line) < 181:
                continue
            try:
                a   = float(line[168:181])
                ecc = float(line[158:168])
                inc = float(line[148:158]) * DEG2RAD
                M0  = float(line[115:125])
                om  = float(line[126:136]) * DEG2RAD
                Om  = float(line[137:147]) * DEG2RAD
            except ValueError:
                continue

            if ecc < 0 or ecc >= 1.0 or a <= 0:
                continue

            name = line[7:25].strip()
            q = a * (1.0 - ecc)

            if q <= 1.3 and a <= 2.5:
                M0_n.append(M0);  om_n.append(om);  Om_n.append(Om)
                inc_n.append(inc); ecc_n.append(ecc); a_n.append(a)
                names_n.append(name)
            elif 3.70 <= a <= 4.20:
                M0_h.append(M0);  om_h.append(om);  Om_h.append(Om)
                inc_h.append(inc); ecc_h.append(ecc); a_h.append(a)
                names_h.append(name)
            elif 4.80 <= a <= 5.50:
                M0_t.append(M0);  om_t.append(om);  Om_t.append(Om)
                inc_t.append(inc); ecc_t.append(ecc); a_t.append(a)
                names_t.append(name)

    n_h, n_t, n_n = len(a_h), len(a_t), len(a_n)
    print(f"  {n_h:,} Hildas, {n_t:,} Trojans, {n_n:,} NEOs in {pytime.time()-t0:.1f}s")

    fam_h = np.full(n_h, FAMILY_HILDA,  dtype=np.int8)
    fam_t = np.full(n_t, FAMILY_TROJAN, dtype=np.int8)
    fam_n = np.full(n_n, FAMILY_NEO,    dtype=np.int8)
    return {
        'M0':    np.concatenate([M0_h,  M0_t,  M0_n]).astype(np.float64),
        'omega': np.concatenate([om_h,  om_t,  om_n]).astype(np.float64),
        'Omega': np.concatenate([Om_h,  Om_t,  Om_n]).astype(np.float64),
        'inc':   np.concatenate([inc_h, inc_t, inc_n]).astype(np.float64),
        'ecc':   np.concatenate([ecc_h, ecc_t, ecc_n]).astype(np.float64),
        'a':     np.concatenate([a_h,   a_t,   a_n]).astype(np.float64),
        'family': np.concatenate([fam_h, fam_t, fam_n]),
        'names': names_h + names_t + names_n,
    }


# ---------------------------------------------------------------------------
# Orbital mechanics (vectorized)
# ---------------------------------------------------------------------------

def compute_positions(a, ecc, inc, omega, Omega, M):
    """
    Given orbital elements and current mean anomaly M (radians),
    compute heliocentric (x, y, z) in AU.  All inputs are NumPy arrays.
    """
    # Solve Kepler's equation E - e sin E = M  (Newton-Raphson, 6 iters)
    E = M.copy()
    for _ in range(6):
        dE = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        E -= dE

    # True anomaly
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    sqrt_fac = np.sqrt(1.0 - ecc * ecc)
    nu = np.arctan2(sqrt_fac * sin_E, cos_E - ecc)

    # Radius
    r = a * (1.0 - ecc * cos_E)

    # Position in orbital plane
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)
    x_orb = r * cos_nu
    y_orb = r * sin_nu

    # Rotation to heliocentric frame
    cos_om = np.cos(omega)
    sin_om = np.sin(omega)
    cos_Om = np.cos(Omega)
    sin_Om = np.sin(Omega)
    cos_i = np.cos(inc)
    sin_i = np.sin(inc)

    # Combined rotation
    x = (cos_Om * cos_om - sin_Om * sin_om * cos_i) * x_orb + \
        (-cos_Om * sin_om - sin_Om * cos_om * cos_i) * y_orb
    y = (sin_Om * cos_om + cos_Om * sin_om * cos_i) * x_orb + \
        (-sin_Om * sin_om + cos_Om * cos_om * cos_i) * y_orb
    z = (sin_om * sin_i) * x_orb + (cos_om * sin_i) * y_orb

    return x, y, z


def orbit_points_3d(a, ecc, inc, omega, Omega, n_pts=ORBIT_SEGMENTS):
    """Return heliocentric (x, y, z) arrays sampled uniformly around one orbit."""
    E = np.linspace(0.0, TWO_PI, n_pts, endpoint=False)
    cos_E = np.cos(E)
    sin_E = np.sin(E)
    sqrt_fac = math.sqrt(1.0 - ecc * ecc)
    nu = np.arctan2(sqrt_fac * sin_E, cos_E - ecc)
    r = a * (1.0 - ecc * cos_E)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    cos_om, sin_om = math.cos(omega), math.sin(omega)
    cos_Om, sin_Om = math.cos(Omega), math.sin(Omega)
    cos_i,  sin_i  = math.cos(inc),   math.sin(inc)
    x = (cos_Om * cos_om - sin_Om * sin_om * cos_i) * x_orb + \
        (-cos_Om * sin_om - sin_Om * cos_om * cos_i) * y_orb
    y = (sin_Om * cos_om + cos_Om * sin_om * cos_i) * x_orb + \
        (-sin_Om * sin_om + cos_Om * cos_om * cos_i) * y_orb
    z = (sin_om * sin_i) * x_orb + (cos_om * sin_i) * y_orb
    return x, y, z


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def world_to_screen(x, z, cam_x_center, cam_x_span, cam_y_span):
    """Convert (x_AU, z_AU) to pixel coordinates for the edge-on view."""
    sx = (x - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH
    sy = HEIGHT / 2 - z / cam_y_span * HEIGHT
    return sx, sy


def depth_color(y, max_depth=6.0):
    """
    Map y-coordinate (depth into screen) to a color.
    Closer (smaller |y|) → brighter; farther → dimmer.
    """
    t = np.clip(np.abs(y) / max_depth, 0.0, 1.0)
    r = (200 + 55 * (1 - t)).astype(np.uint8)
    g = (180 + 60 * (1 - t)).astype(np.uint8)
    b = (140 + 115 * (1 - t)).astype(np.uint8)
    return r, g, b


def depth_color_hilda(y, max_depth=6.0):
    """Amber-orange for Hildas (C/D/P-type colour scheme)."""
    t = np.clip(np.abs(y) / max_depth, 0.0, 1.0)
    r = (220 + 35 * (1 - t)).astype(np.uint8)
    g = (130 + 50 * (1 - t)).astype(np.uint8)
    b = ( 30 + 30 * (1 - t)).astype(np.uint8)
    return r, g, b


def depth_color_trojan(y, max_depth=6.0):
    """Brick-red for Jupiter Trojans (D-type colour scheme)."""
    t = np.clip(np.abs(y) / max_depth, 0.0, 1.0)
    r = (190 + 50 * (1 - t)).astype(np.uint8)
    g = ( 70 + 50 * (1 - t)).astype(np.uint8)
    b = ( 50 + 30 * (1 - t)).astype(np.uint8)
    return r, g, b


def depth_color_neo(y, max_depth=6.0):
    """Bright cyan-green for NEOs (high-visibility warning colour)."""
    t = np.clip(np.abs(y) / max_depth, 0.0, 1.0)
    r = ( 40 + 30 * (1 - t)).astype(np.uint8)
    g = (220 + 35 * (1 - t)).astype(np.uint8)
    b = (160 + 60 * (1 - t)).astype(np.uint8)
    return r, g, b


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    table2_path = "data/table2.dat.gz"
    astorb_path = "data/astorb.dat.gz"
    if len(sys.argv) > 1:
        table2_path = sys.argv[1]
    if len(sys.argv) > 2:
        astorb_path = sys.argv[2]

    # Load data
    data = load_table2_asteroids(table2_path, astorb_path, MAX_ASTEROIDS)
    if len(data['a']) == 0:
        print("No asteroids loaded. Check data path.")
        sys.exit(1)

    # Tag main-belt asteroids, then merge Hilda + Trojan families from astorb
    N_main = len(data['a'])
    data['family'] = np.zeros(N_main, dtype=np.int8)
    families = load_families_from_astorb(astorb_path)
    for key in ('M0', 'omega', 'Omega', 'inc', 'ecc', 'a', 'family'):
        data[key] = np.concatenate([data[key], families[key]])
    data['names'] = data['names'] + families['names']

    N = len(data['a'])
    N_hilda  = int(np.sum(data['family'] == FAMILY_HILDA))
    N_trojan = int(np.sum(data['family'] == FAMILY_TROJAN))
    N_neo    = int(np.sum(data['family'] == FAMILY_NEO))
    print(f"Total: {N:,}  (main-belt {N_main:,}, Hildas {N_hilda:,}, Trojans {N_trojan:,}, NEOs {N_neo:,})")

    # Pre-compute mean motion  n = 2π / P,  P = a^(3/2) years
    periods = data['a'] ** 1.5              # orbital period in years
    mean_motion = TWO_PI / periods          # rad/year

    # Orbital speed: years of orbital time per real second.
    # At 0.5 yr/s, an inner-belt asteroid (P ≈ 4 yr) completes one orbit in ~8s.
    ORBITAL_SPEED = 0.5  # years per real-second

    # Per-asteroid accumulated orbital phase (radians), independent of sim_time
    M0_rad = data['M0'] * DEG2RAD           # initial M₀ in radians
    M_phase = np.zeros(N, dtype=np.float64) # accumulated orbital advance

    # Pre-compute planet 3D orbit points (fixed; camera transform applied per frame)
    planet_orbits = []
    for pl_name, pl_a, pl_ecc, pl_inc_d, pl_om_d, pl_Om_d, pl_color in PLANETS:
        px, py, pz = orbit_points_3d(
            pl_a, pl_ecc,
            pl_inc_d * DEG2RAD, pl_om_d * DEG2RAD, pl_Om_d * DEG2RAD,
        )
        planet_orbits.append((pl_name, pl_color, px, py, pz))

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Asteroid Belt — Edge-on View")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16, bold=True)

    # Camera state
    cam_x_center = DEFAULT_X_CENTER
    cam_x_span = DEFAULT_X_SPAN
    cam_azimuth = 0.0                       # rotation around vertical (z) axis, radians
    cam_elevation = 0.0                     # tilt up/down (rotation around camera x-axis), radians
    cam_roll = 0.0                          # roll (rotation around camera depth axis), radians
    cam_y_offset = 0.0                      # vertical pan offset (AU)

    # Time state
    sim_time = 0.0              # simulation time in "degrees" (activation clock only)
    time_speed = TIME_SPEED_DEFAULT
    paused = False
    reverse = False                         # F key toggles direction
    show_hud = True                         # H key toggles HUD visibility
    show_orbits = True                      # O key toggles planet orbit lines

    running = True
    while running:
        dt_real = clock.tick(FPS_CAP) / 1000.0  # real seconds elapsed
        # Uniform AU-to-pixel scale: derive vertical span from horizontal span
        cam_y_span = cam_x_span * HEIGHT / WIDTH

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim_time = 0.0
                    M_phase[:] = 0.0
                    reverse = False
                elif event.key == pygame.K_f:
                    reverse = not reverse
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_o:
                    show_orbits = not show_orbits
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 0.9 if event.y > 0 else 1.1
                cam_x_span *= zoom_factor

        # Continuous key handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            time_speed = min(time_speed * 1.02, TIME_SPEED_MAX)
        if keys[pygame.K_MINUS]:
            time_speed = max(time_speed * 0.98, TIME_SPEED_MIN)
        if keys[pygame.K_LEFT]:
            cam_x_center -= 0.01 * cam_x_span
        if keys[pygame.K_RIGHT]:
            cam_x_center += 0.01 * cam_x_span
        if keys[pygame.K_UP]:
            cam_y_offset += 0.01 * cam_y_span   # pan up
        if keys[pygame.K_DOWN]:
            cam_y_offset -= 0.01 * cam_y_span   # pan down
        # Camera rotation
        if keys[pygame.K_a]:
            cam_azimuth -= 0.01  # rotate left
        if keys[pygame.K_d]:
            cam_azimuth += 0.01  # rotate right
        if keys[pygame.K_w]:
            cam_elevation = min(cam_elevation + 0.01, math.pi / 2)  # tilt up
        if keys[pygame.K_s]:
            cam_elevation = max(cam_elevation - 0.01, -math.pi / 2)  # tilt down
        if keys[pygame.K_z]:
            cam_roll -= 0.01  # roll left
        if keys[pygame.K_x]:
            cam_roll += 0.01  # roll right

        # --- Update ---
        if not paused:
            if reverse:
                sim_time = max(0.0, sim_time - time_speed * dt_real)
            else:
                sim_time += time_speed * dt_real

            # Moving asteroids: M₀ ≤ sim_time
            moving_mask = data['M0'] <= sim_time
            dM = mean_motion * dt_real * ORBITAL_SPEED

            if reverse:
                # Orbits run backward for still-moving asteroids
                M_phase -= np.where(moving_mask, dM, 0.0)
                M_phase = np.maximum(M_phase, 0.0)
                # Reset phase for asteroids that just stopped
                M_phase = np.where(moving_mask, M_phase, 0.0)
            else:
                # Advance orbital phase for moving asteroids
                M_phase += np.where(moving_mask, dM, 0.0)
        else:
            moving_mask = data['M0'] <= sim_time

        n_moving = int(np.sum(moving_mask))

        # --- Compute positions ---
        # Moving asteroids: real 3D orbital positions
        M_current = (M0_rad + M_phase) % TWO_PI
        x, y, z = compute_positions(data['a'], data['ecc'], data['inc'],
                                    data['omega'], data['Omega'], M_current)

        # Waiting asteroids: flat (a, i) plane display.
        # x = a (semi-major axis), y = 0, z = inclination (in degrees, scaled to AU)
        inc_deg = data['inc'] / DEG2RAD     # inclination in degrees
        i_scale = cam_y_span / 60.0         # scale: 60° fills the vertical span
        waiting_mask = ~moving_mask
        x = np.where(waiting_mask, data['a'], x)
        y = np.where(waiting_mask, 0.0, y)
        z = np.where(waiting_mask, inc_deg * i_scale, z)

        # --- Camera rotation ---
        # 1. Azimuth: rotate around vertical (z) axis
        cos_az = math.cos(cam_azimuth)
        sin_az = math.sin(cam_azimuth)
        x1 = x * cos_az - y * sin_az
        y1 = x * sin_az + y * cos_az
        z1 = z

        # 2. Elevation: tilt camera up/down (rotate around camera x-axis)
        cos_el = math.cos(cam_elevation)
        sin_el = math.sin(cam_elevation)
        y_cam = y1 * cos_el - z1 * sin_el   # depth
        z2 = y1 * sin_el + z1 * cos_el      # screen vertical
        x2 = x1                               # screen horizontal

        # 3. Roll: rotate around camera depth axis (y_cam)
        cos_rl = math.cos(cam_roll)
        sin_rl = math.sin(cam_roll)
        x_cam = x2 * cos_rl - z2 * sin_rl
        z_cam = x2 * sin_rl + z2 * cos_rl

        # --- Project to screen ---
        scr_x = (x_cam - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH
        scr_y = HEIGHT / 2 - (z_cam - cam_y_offset) / cam_y_span * HEIGHT

        # Filter to visible
        visible = ((scr_x >= 0) & (scr_x < WIDTH) &
                   (scr_y >= 0) & (scr_y < HEIGHT))
        scr_x_vis  = scr_x[visible].astype(np.int32)
        scr_y_vis  = scr_y[visible].astype(np.int32)
        y_cam_vis  = y_cam[visible]
        family_vis = data['family'][visible]
        n_visible  = len(scr_x_vis)

        # --- Draw ---
        screen.fill(BG_COLOR)

        # Background gradient (subtle)
        for row in range(0, HEIGHT, 4):
            t = row / HEIGHT
            c = int(4 + 8 * (1 - abs(t - 0.5) * 2))
            pygame.draw.line(screen, (c, c, c + 4), (0, row), (WIDTH, row))

        # Draw Sun (apply same camera transform as asteroids)
        # Sun is at origin (0, 0, 0) — after camera rotation it stays at origin
        sun_x_cam = 0.0
        sun_z_cam = 0.0
        sun_sx = int((sun_x_cam - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH)
        sun_sy = int(HEIGHT / 2 - (sun_z_cam - cam_y_offset) / cam_y_span * HEIGHT)
        if -20 < sun_sx < WIDTH + 20:
            pygame.draw.circle(screen, SUN_COLOR, (sun_sx, sun_sy), 8)
            # Glow
            glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
            for r in range(20, 0, -1):
                alpha = int(60 * (1 - r / 20))
                pygame.draw.circle(glow_surf, (*SUN_COLOR, alpha), (20, 20), r)
            screen.blit(glow_surf, (sun_sx - 20, sun_sy - 20))

        # Draw planet orbits
        if show_orbits:
            for pl_name, pl_color, px, py, pz in planet_orbits:
                # Apply same camera rotation as asteroids
                px1 = px * cos_az - py * sin_az
                py1 = px * sin_az + py * cos_az
                py_cam_p = py1 * cos_el - pz * sin_el
                pz2      = py1 * sin_el + pz * cos_el
                px_cam   = px1 * cos_rl - pz2 * sin_rl
                pz_cam   = px1 * sin_rl + pz2 * cos_rl
                pscr_x = (px_cam - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH
                pscr_y = HEIGHT / 2 - (pz_cam - cam_y_offset) / cam_y_span * HEIGHT
                pts = [(int(pscr_x[i]), int(pscr_y[i])) for i in range(len(pscr_x))]
                pygame.draw.lines(screen, pl_color, True, pts, 1)
                # Label at rightmost on-screen point
                on_screen = [(i, pscr_x[i]) for i in range(len(pscr_x))
                             if 0 <= pscr_x[i] < WIDTH and 0 <= pscr_y[i] < HEIGHT]
                if on_screen:
                    best = max(on_screen, key=lambda t: t[1])[0]
                    lbl = font.render(pl_name, True, pl_color)
                    screen.blit(lbl, (int(pscr_x[best]) + 4, int(pscr_y[best]) - 14))

        # Draw asteroids (colour-coded by family)
        if n_visible > 0:
            pxarray = pygame.surfarray.pixels3d(screen)
            ix = np.clip(scr_x_vis, 0, WIDTH - 1)
            iy = np.clip(scr_y_vis, 0, HEIGHT - 1)
            for fam_id, color_fn in (
                (FAMILY_MAIN_BELT, depth_color),
                (FAMILY_HILDA,     depth_color_hilda),
                (FAMILY_TROJAN,    depth_color_trojan),
                (FAMILY_NEO,       depth_color_neo),
            ):
                fmask = family_vis == fam_id
                if fmask.any():
                    rc, gc, bc = color_fn(y_cam_vis[fmask])
                    pxarray[ix[fmask], iy[fmask], 0] = rc
                    pxarray[ix[fmask], iy[fmask], 1] = gc
                    pxarray[ix[fmask], iy[fmask], 2] = bc
            del pxarray  # release surface lock

        # --- HUD ---
        if show_hud:
            fps = clock.get_fps()
            activation_pct = min(sim_time / 360.0 * 100, 100) if sim_time <= 360 else 100
            az_deg = math.degrees(cam_azimuth) % 360
            el_deg = math.degrees(cam_elevation)
            rl_deg = math.degrees(cam_roll) % 360

            direction = "\u25c0 REVERSE" if reverse else "\u25b6 FORWARD"
            hud_lines = [
                f"FPS: {fps:5.1f}   {direction}",
                f"Time: {sim_time:8.1f}\u00b0  Speed: {time_speed:6.1f}\u00b0/s",
                f"Moving: {n_moving:,} / {N:,}  ({activation_pct:.0f}%)",
                f"Visible: {n_visible:,}   Belt:{N_main:,}  Hilda:{N_hilda:,}  Trojan:{N_trojan:,}  NEO:{N_neo:,}",
                f"Zoom: {cam_x_span:.2f} AU x {cam_y_span:.2f} AU",
                f"Az: {az_deg:.1f}\u00b0  El: {el_deg:.1f}\u00b0  Ecl: {rl_deg:.1f}\u00b0",
            ]
            if paused:
                hud_lines.append("** PAUSED **")

            y_pos = 10
            for line in hud_lines:
                surf = font.render(line, True, HUD_COLOR)
                screen.blit(surf, (10, y_pos))
                y_pos += 20

            # Family colour legend (top-right)
            legend = [
                ("Main belt", (220, 200, 200)),
                ("Hildas",    (255, 180,  60)),
                ("Trojans",   (240, 120,  80)),
                ("NEOs",      ( 60, 255, 180)),
            ]
            lx, ly = WIDTH - 140, 10
            for lname, lcol in legend:
                pygame.draw.circle(screen, lcol, (lx, ly + 7), 4)
                screen.blit(font.render(lname, True, lcol), (lx + 12, ly))
                ly += 20

        # Controls help (bottom)
        controls = "SPACE=pause  F=reverse  +/-=speed  arrows=pan  A/D=azimuth  W/S=elevation  Z/X=ecliptic  scroll=zoom  H=HUD  O=orbits  R=reset  Q=quit"
        ctrl_surf = font.render(controls, True, (100, 100, 120))
        screen.blit(ctrl_surf, (10, HEIGHT - 25))

        # AU scale markers along bottom
        au_start = max(0, int(cam_x_center - cam_x_span / 2))
        au_end = int(cam_x_center + cam_x_span / 2) + 1
        for au in range(au_start, au_end + 1):
            sx_au = int((au - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH)
            if 0 <= sx_au < WIDTH:
                pygame.draw.line(screen, (40, 40, 50), (sx_au, HEIGHT - 35),
                                 (sx_au, HEIGHT - 30))
                label = font.render(f"{au} AU", True, (60, 60, 80))
                screen.blit(label, (sx_au - 15, HEIGHT - 50))

        pygame.display.flip()

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()

