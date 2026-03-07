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
DEFAULT_Y_CENTER = 1.0              # AU — vertical centre of view
DEFAULT_X_SPAN = 8.0                # AU — total horizontal span
# cam_y_span is derived each frame as cam_x_span * HEIGHT / WIDTH (uniform scale)

# Time: simulation time in "seconds" where each second = 1° of M₀ activation
TIME_SPEED_DEFAULT = 30.0           # degrees / real-second
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
# Camera animation timeline
# ---------------------------------------------------------------------------
# Toggle at runtime with the K key.  Set CAMERA_ANIM_ENABLED = True to start
# with animation active.
#
# Each entry in CAMERA_TIMELINE is a keyframe dict describing the ABSOLUTE
# camera state at real-time 't' (seconds from animation start / last K press).
# Any field that is omitted falls back to the value in _CAM_DEFAULTS.
#
# The 'ease' key controls the transition FROM this keyframe TO the next one:
#   'linear'      — constant speed
#   'smooth'      — smoothstep: slow-in / slow-out  (default)
#   'ease_in'     — accelerates toward the next keyframe
#   'ease_out'    — decelerates into the next keyframe
#   'ease_in_out' — cubic: slow → fast → slow
#
# Animatable fields:
#   x_center   (AU)   horizontal centre of view
#   x_span     (AU)   visible width  /  zoom level
#   azimuth    (deg)  rotation around ecliptic-north axis
#   elevation  (deg)  camera tilt  (0 = edge-on, 90 = top-down, −90 = bottom-up)
#   roll       (deg)  twist around the view direction
#   y_center   (AU)   vertical centre of view
#   time_speed (°/s)  asteroid-activation time speed
# ---------------------------------------------------------------------------

CAMERA_ANIM_ENABLED = False

CAMERA_TIMELINE = [
    # t=0  : start edge-on, belt centred
    {
        't': 0.0,
        'x_center': DEFAULT_X_CENTER, 'x_span': 8.0,
        'azimuth': 0.0, 'elevation': 0.0, 'roll': 0.0, 'y_center': DEFAULT_Y_CENTER,
        'time_speed': 10.0,
        'ease': 'smooth',
    },
    # t=8  : tilt up to top-down view
    {
        't': 8.0,
        'x_center': DEFAULT_X_CENTER, 'x_span': 8.0,
        'azimuth': 0.0, 'elevation': 90.0, 'roll': 0.0, 'y_center': DEFAULT_Y_CENTER,
        'time_speed': 10.0,
        'ease': 'smooth',
    },
    # t=16 : zoom out and swing the camera to an oblique angle
    {
        't': 16.0,
        'x_center': 4.0, 'x_span': 14.0,
        'azimuth': 30.0, 'elevation': 45.0, 'roll': 0.0, 'y_center': DEFAULT_Y_CENTER,
        'time_speed': 20.0,
        'ease': 'ease_out',
    },
    # t=28 : return to default edge-on view
    {
        't': 28.0,
        'x_center': DEFAULT_X_CENTER, 'x_span': 8.0,
        'azimuth': 0.0, 'elevation': 0.0, 'roll': 0.0, 'y_center': DEFAULT_Y_CENTER,
        'time_speed': 10.0,
        'ease': 'smooth',
    },
]

_CAM_DEFAULTS = {
    'x_center':   DEFAULT_X_CENTER,
    'x_span':     8.0,
    'azimuth':    0.0,
    'elevation':  0.0,
    'roll':       0.0,
    'y_center':   DEFAULT_Y_CENTER,
    'time_speed': TIME_SPEED_DEFAULT,
}


def _cam_interp(t: float) -> dict:
    """
    Interpolate camera parameters at real-time t (seconds since animation start).
    Returns a dict with the same keys as _CAM_DEFAULTS (angles in degrees).
    """
    kf = CAMERA_TIMELINE
    if not kf:
        return dict(_CAM_DEFAULTS)

    def _kf_vals(k):
        return {field: k.get(field, _CAM_DEFAULTS[field]) for field in _CAM_DEFAULTS}

    if t <= kf[0]['t']:
        return _kf_vals(kf[0])
    if t >= kf[-1]['t']:
        return _kf_vals(kf[-1])

    # Locate surrounding keyframes
    k0, k1 = kf[-2], kf[-1]
    for i in range(len(kf) - 1):
        if kf[i]['t'] <= t < kf[i + 1]['t']:
            k0, k1 = kf[i], kf[i + 1]
            break

    dt_kf = k1['t'] - k0['t']
    alpha = (t - k0['t']) / dt_kf if dt_kf > 0 else 1.0

    ease = k0.get('ease', 'smooth')
    if ease == 'smooth':
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    elif ease == 'ease_in':
        alpha = alpha * alpha
    elif ease == 'ease_out':
        alpha = 1.0 - (1.0 - alpha) ** 2
    elif ease == 'ease_in_out':
        if alpha < 0.5:
            alpha = 4.0 * alpha ** 3
        else:
            alpha = 1.0 - (-2.0 * alpha + 2.0) ** 3 / 2.0
    # 'linear': alpha unchanged

    return {
        field: k0.get(field, _CAM_DEFAULTS[field]) * (1.0 - alpha)
               + k1.get(field, _CAM_DEFAULTS[field]) * alpha
        for field in _CAM_DEFAULTS
    }


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def load_asteroids(astorb_path: str, table2_path: str,
                  max_count: int = MAX_ASTEROIDS):
    """
    Load up to max_count asteroids from astorb.dat(.gz), upgrading a/e/i to
    proper elements from table2.dat.gz where available.

    Steps:
      1. Read proper elements from table2.dat.gz into a lookup keyed by
         unpacked asteroid number.
      2. Stream astorb.dat up to max_count records.  For each asteroid use
         proper a, e, i when the lookup contains a match; otherwise keep
         the osculating a, e, i.  M0, omega, Omega always come from astorb.
      3. Classify family from the final a/e:
           NEO       : q = a(1-e) <= 1.3 AU  and  a <= 1.5 AU
           Hilda     : 3.70 <= a <= 4.20 AU
           Trojan    : 4.80 <= a <= 5.50 AU
           Main belt : everything else

    astorb.dat fixed-width columns (0-indexed):
      number [0:6]  name [7:25]  M0 [115:125]  omega [126:136]  Omega [137:147]
      inc [148:158]  ecc [158:168]  a [168:181]

    table2.dat whitespace-separated, 12 fields per line:
      col 0: ap(AU)  col 2: ep  col 4: sinIp  col 11: unpacked number
    """
    # --- Step 1: proper elements lookup ---
    proper = {}  # unpacked_number -> (a_p, e_p, i_p_rad)
    opener2 = gzip.open if table2_path.endswith('.gz') else open
    print(f"Reading proper elements from {table2_path}...")
    t0 = pytime.time()
    with opener2(table2_path, 'rt', encoding='ascii', errors='replace') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 12:
                continue
            try:
                a_p  = float(parts[0])
                e_p  = float(parts[2])
                sinI = float(parts[4])
                key  = parts[11].strip()
            except ValueError:
                continue
            if a_p <= 0 or e_p < 0 or e_p >= 1.0 or abs(sinI) > 1.0:
                continue
            proper[key] = (a_p, e_p, math.asin(min(max(sinI, -1.0), 1.0)))
    print(f"  {len(proper):,} proper-element records in {pytime.time()-t0:.1f}s")

    # --- Step 2: stream astorb ---
    opener1 = gzip.open if astorb_path.endswith('.gz') else open
    print(f"Loading asteroids from {astorb_path}...")
    t0 = pytime.time()

    M0_list     = []
    omega_list  = []
    Omega_list  = []
    inc_list    = []
    ecc_list    = []
    a_list      = []
    family_list = []
    names       = []
    n_proper    = 0

    with opener1(astorb_path, 'rt', encoding='ascii', errors='replace') as f:
        for line in f:
            if len(line) < 181:
                continue
            num_str = line[0:6].strip()
            if not num_str:
                continue
            try:
                M0  = float(line[115:125])
                om  = float(line[126:136]) * DEG2RAD
                Om  = float(line[137:147]) * DEG2RAD
                i_o = float(line[148:158]) * DEG2RAD
                e_o = float(line[158:168])
                a_o = float(line[168:181])
            except ValueError:
                continue
            if e_o < 0 or e_o >= 1.0 or a_o <= 0:
                continue

            if num_str in proper:
                a, e, inc = proper[num_str]
                n_proper += 1
            else:
                a, e, inc = a_o, e_o, i_o

            q = a * (1.0 - e)
            if q <= 1.3 and a <= 1.5:
                fam = FAMILY_NEO
            elif 3.70 <= a <= 4.20:
                fam = FAMILY_HILDA
            elif 4.80 <= a <= 5.50:
                fam = FAMILY_TROJAN
            else:
                fam = FAMILY_MAIN_BELT

            M0_list.append(M0)
            omega_list.append(om)
            Omega_list.append(Om)
            inc_list.append(inc)
            ecc_list.append(e)
            a_list.append(a)
            family_list.append(fam)
            names.append(line[7:25].strip())

            if len(a_list) >= max_count:
                break

    N = len(a_list)
    print(f"  {N:,} asteroids ({n_proper:,} with proper elements) in {pytime.time()-t0:.1f}s")

    return {
        'M0':    np.array(M0_list,    dtype=np.float64),
        'omega': np.array(omega_list, dtype=np.float64),
        'Omega': np.array(Omega_list, dtype=np.float64),
        'inc':   np.array(inc_list,   dtype=np.float64),
        'ecc':   np.array(ecc_list,   dtype=np.float64),
        'a':     np.array(a_list,     dtype=np.float64),
        'family': np.array(family_list, dtype=np.int8),
        'names': names,
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
    data = load_asteroids(astorb_path, table2_path, MAX_ASTEROIDS)
    N = len(data['a'])
    if N == 0:
        print("No asteroids loaded. Check data path.")
        sys.exit(1)

    N_main   = int(np.sum(data['family'] == FAMILY_MAIN_BELT))
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

    # Per-asteroid accumulated orbital phase (radians).
    # Initialised to the y=0 crossing so every asteroid starts in the
    # screen plane (no depth offset) when it activates.
    # y=0 when: sin(Ω)cos(ω+ν) + cos(Ω)cos(i)sin(ω+ν) = 0
    # => ω+ν = arctan2(-sin(Ω), cos(Ω)·cos(i))
    u_y0 = np.arctan2(-np.sin(data['Omega']),
                       np.cos(data['Omega']) * np.cos(data['inc']))
    nu_y0 = (u_y0 - data['omega']) % TWO_PI
    # True anomaly → eccentric anomaly → mean anomaly
    sqrt_fac_node = np.sqrt((1.0 - data['ecc']) / (1.0 + data['ecc']))
    E_node = 2.0 * np.arctan2(sqrt_fac_node * np.sin(nu_y0 / 2.0),
                               np.cos(nu_y0 / 2.0))
    M_node = (E_node - data['ecc'] * np.sin(E_node)) % TWO_PI
    M_phase = M_node.copy()

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

    # Motion trail: persistent float buffer faded each frame, blitted additively
    trail_pixels = np.zeros((WIDTH, HEIGHT, 3), dtype=np.float32)
    trail_surf = pygame.Surface((WIDTH, HEIGHT))
    trail_surf.fill((0, 0, 0))

    # Camera state
    cam_x_center = DEFAULT_X_CENTER
    cam_x_span = DEFAULT_X_SPAN
    cam_azimuth = 0.0                       # rotation around vertical (z) axis, radians
    cam_elevation = 0.0                     # tilt up/down (rotation around camera x-axis), radians
    cam_roll = 0.0                          # roll (rotation around camera depth axis), radians
    cam_y_center = DEFAULT_Y_CENTER         # vertical centre of view (AU)
    cam_anim_enabled = CAMERA_ANIM_ENABLED  # K key toggles
    cam_anim_time = 0.0                     # real seconds since animation start / last K press

    # Time state
    sim_time = 0.0              # simulation time in "degrees" (activation clock only)
    time_speed = TIME_SPEED_DEFAULT
    paused = False
    reverse = False                         # F key toggles direction
    show_hud = True                         # H key toggles HUD visibility
    show_orbits = True                      # O key toggles planet orbit lines
    show_ai_plane = True                    # I key toggles: a/i plane vs real positions

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
                    M_phase[:] = M_node
                    reverse = False
                    cam_anim_time = 0.0
                    trail_pixels[:] = 0.0
                    trail_surf.fill((0, 0, 0))
                elif event.key == pygame.K_f:
                    reverse = not reverse
                elif event.key == pygame.K_h:
                    show_hud = not show_hud
                elif event.key == pygame.K_o:
                    show_orbits = not show_orbits
                elif event.key == pygame.K_i:
                    show_ai_plane = not show_ai_plane
                elif event.key == pygame.K_k:
                    cam_anim_enabled = not cam_anim_enabled
                    cam_anim_time = 0.0   # restart timeline from first keyframe
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 0.9 if event.y > 0 else 1.1
                cam_x_span *= zoom_factor

        # Continuous key handling
        keys = pygame.key.get_pressed()
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            time_speed = min(time_speed * 1.02, TIME_SPEED_MAX)
        if keys[pygame.K_MINUS]:
            time_speed = max(time_speed * 0.98, TIME_SPEED_MIN)
        if not cam_anim_enabled:
            if keys[pygame.K_LEFT]:
                cam_x_center -= 0.01 * cam_x_span
            if keys[pygame.K_RIGHT]:
                cam_x_center += 0.01 * cam_x_span
            if keys[pygame.K_UP]:
                cam_y_center += 0.01 * cam_y_span   # pan up
            if keys[pygame.K_DOWN]:
                cam_y_center -= 0.01 * cam_y_span   # pan down
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
                # Don't wind back past the starting ascending-node angle
                M_phase = np.maximum(M_phase, M_node)
                # Reset phase for asteroids that just stopped
                M_phase = np.where(moving_mask, M_phase, M_node)
            else:
                # Advance orbital phase for moving asteroids
                M_phase += np.where(moving_mask, dM, 0.0)
        else:
            moving_mask = data['M0'] <= sim_time

        n_moving = int(np.sum(moving_mask))

        # --- Camera animation ---
        cam_anim_time += dt_real
        if cam_anim_enabled:
            _cs = _cam_interp(cam_anim_time)
            cam_x_center  = _cs['x_center']
            cam_x_span    = _cs['x_span']
            cam_azimuth   = math.radians(_cs['azimuth'])
            cam_elevation = math.radians(_cs['elevation'])
            cam_roll      = math.radians(_cs['roll'])
            cam_y_center  = _cs['y_center']
            time_speed    = max(TIME_SPEED_MIN, min(_cs['time_speed'], TIME_SPEED_MAX))
            cam_y_span    = cam_x_span * HEIGHT / WIDTH   # recompute after override

        # --- Compute positions ---
        # Moving asteroids: real 3D orbital positions starting from y=0.
        # Waiting asteroids: a/i plane — x=a, y=0, z=inclination (scaled).
        M_current = M_phase % TWO_PI
        x, y, z = compute_positions(data['a'], data['ecc'], data['inc'],
                                    data['omega'], data['Omega'], M_current)

        # Waiting asteroids: a/i distribution display (I=toggle) or real frozen positions.
        # z = a·sin(i) is the physical max orbital height in AU — same units as the
        # real orbit view, so the scale matches when toggling with I.
        waiting_mask = ~moving_mask
        if show_ai_plane:
            x = np.where(waiting_mask, data['a'], x)
            y = np.where(waiting_mask, 0.0, y)
            z = np.where(waiting_mask, data['a'] * np.sin(data['inc']), z)
        # else: keep real orbital positions frozen at M_node (y=0 crossing)

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
        scr_y = HEIGHT / 2 - (z_cam - cam_y_center) / cam_y_span * HEIGHT

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

        # Motion trail: decay, update with current moving-asteroid positions, blit
        trail_pixels *= 0.66  # decay existing trail
        if n_visible > 0:
            ix_t = np.clip(scr_x_vis, 0, WIDTH - 1)
            iy_t = np.clip(scr_y_vis, 0, HEIGHT - 1)
            moving_vis = moving_mask[visible]
            if moving_vis.any():
                ix_mov = ix_t[moving_vis]
                iy_mov = iy_t[moving_vis]
                y_cam_mov = y_cam_vis[moving_vis]
                fam_mov = family_vis[moving_vis]
                for fam_id, color_fn in (
                    (FAMILY_MAIN_BELT, depth_color),
                    (FAMILY_HILDA,     depth_color_hilda),
                    (FAMILY_TROJAN,    depth_color_trojan),
                    (FAMILY_NEO,       depth_color_neo),
                ):
                    fm = fam_mov == fam_id
                    if fm.any():
                        rc, gc, bc = color_fn(y_cam_mov[fm])
                        trail_pixels[ix_mov[fm], iy_mov[fm], 0] = rc
                        trail_pixels[ix_mov[fm], iy_mov[fm], 1] = gc
                        trail_pixels[ix_mov[fm], iy_mov[fm], 2] = bc
        trail_arr = pygame.surfarray.pixels3d(trail_surf)
        np.copyto(trail_arr, trail_pixels.clip(0, 255).astype(np.uint8))
        del trail_arr
        screen.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_ADD)

        # Draw Sun (apply same camera transform as asteroids)
        # Sun is at origin (0, 0, 0) — after camera rotation it stays at origin
        sun_x_cam = 0.0
        sun_z_cam = 0.0
        sun_sx = int((sun_x_cam - (cam_x_center - cam_x_span / 2)) / cam_x_span * WIDTH)
        sun_sy = int(HEIGHT / 2 - (sun_z_cam - cam_y_center) / cam_y_span * HEIGHT)
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
                pscr_y = HEIGHT / 2 - (pz_cam - cam_y_center) / cam_y_span * HEIGHT
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
            if cam_anim_enabled and CAMERA_TIMELINE:
                total_t = CAMERA_TIMELINE[-1]['t']
                anim_pct = min(cam_anim_time / total_t * 100, 100) if total_t > 0 else 100
                hud_lines.append(f"CamAnim: {cam_anim_time:.1f}s / {total_t:.0f}s  ({anim_pct:.0f}%)  K=off")
            else:
                hud_lines.append("CamAnim: OFF  (K=on to start)")
            if paused:
                hud_lines.append("** PAUSED **")
            hud_lines.append(f"Waiting: {'a/i plane' if show_ai_plane else 'real orbit'}  (I=toggle)")

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
        controls = "SPACE=pause  F=reverse  +/-=speed  arrows=pan  A/D=azimuth  W/S=elevation  Z/X=ecliptic  scroll=zoom  H=HUD  O=orbits  I=a/i  K=anim  R=reset  Q=quit"
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

