"""Bridge inspection geometry helpers extending GCBF+ under the MIT license."""

import jax
import jax.numpy as jnp
import numpy as np

DECK_Z = 0.0
N_ZONES = 48
CABLE_ZONE_HEIGHTS = np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0], dtype=np.float32)
CABLE_ZONE_OFFSET = 3.0
PYLON_INSPECT_OFFSET = 3.5
ZONE_VISIT_RADIUS = 2.5

# ─── SECTION A: Pure Python / numpy — reset-time ONLY ─────────
def get_zone_centroids(bridge_length, bridge_width, pylon_xy) -> np.ndarray:
    """Returns all inspection zone centroids.

    Args:
        bridge_length: Scalar float bridge span length in metres.
        bridge_width: Scalar float deck width in metres.
        pylon_xy: (2, 2) float array of pylon xy positions in metres.

    Returns:
        (48, 3) float32 numpy array of zone centroids in metres.
    """
    xs = np.linspace(5.0, float(bridge_length) - 5.0, 4, dtype=np.float32)
    ys = np.linspace(2.0, float(bridge_width) - 2.0, 4, dtype=np.float32)
    deck = np.array([[x, y, DECK_Z + 0.5] for x in xs for y in ys], dtype=np.float32)
    pylons = np.asarray(pylon_xy, dtype=np.float32)
    cable = []
    for px, py in pylons:
        for z in CABLE_ZONE_HEIGHTS:
            cable.append([px + CABLE_ZONE_OFFSET, py, z])
    pylon_zones = []
    for px, py in pylons:
        for idx, z in enumerate(CABLE_ZONE_HEIGHTS):
            sign = 1.0 if idx % 2 == 0 else -1.0
            pylon_zones.append([px + sign * PYLON_INSPECT_OFFSET, py, z])
    zones = np.concatenate([deck, np.asarray(cable, dtype=np.float32), np.asarray(pylon_zones, dtype=np.float32)], axis=0)
    if zones.shape != (N_ZONES, 3):
        raise ValueError(f"Expected {(N_ZONES, 3)} zones, got {zones.shape}")
    return zones.astype(np.float32)


def get_initial_worker_positions(n_workers, zone_centroids, rng_seed) -> np.ndarray:
    """Place workers at randomly selected deck zone centroids.

    Args:
        n_workers: Python int number of active workers.
        zone_centroids: (48, 3) float32 numpy array of zone centroids.
        rng_seed: Python int numpy RNG seed.

    Returns:
        (n_workers, 3) float32 numpy array of worker positions in metres.
    """
    rng = np.random.default_rng(int(rng_seed))
    idx = rng.choice(np.arange(16), size=int(n_workers), replace=False)
    positions = np.asarray(zone_centroids[idx], dtype=np.float32).copy()
    positions[:, 2] = DECK_Z
    return positions

# ─── SECTION B: JAX-jittable — inner loop ONLY ─────────────────
@jax.jit
def check_zone_coverage(drone_positions, zone_centroids, current_coverage, visit_radius) -> jnp.ndarray:
    """Update zone coverage mask.

    Args:
        drone_positions: (n_drones, 3) float32 drone positions in metres.
        zone_centroids: (n_zones, 3) float32 zone positions in metres.
        current_coverage: (n_zones,) bool existing coverage mask.
        visit_radius: Scalar float visit radius in metres.

    Returns:
        (n_zones,) bool updated monotone coverage mask.
    """
    dists = jnp.sqrt(jnp.sum((drone_positions[:, None, :] - zone_centroids[None, :, :]) ** 2, axis=-1) + 1e-8)
    newly_covered = jnp.any(dists < visit_radius, axis=0)
    return jnp.logical_or(current_coverage, newly_covered)
