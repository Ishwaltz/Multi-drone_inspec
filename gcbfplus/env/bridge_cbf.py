"""Pure JAX CBF metrics for the bridge inspection GCBF+ extension."""

import jax
import jax.numpy as jnp

R2_EXCLUSION = 1.5
H2_EXCLUSION = 2.3
PYLON_RADIUS = 2.0
MAX_WORKERS = 5
DT = 0.1
R_SAFE_DRONE = 2.0


@jax.jit
def h_human_radial(drone_xy, worker_xy, R2=R2_EXCLUSION):
    """Radial CBF for human exclusion cylinder.

    Args:
        drone_xy: (2,) float32 drone xy position in metres.
        worker_xy: (2,) float32 worker xy position in metres.
        R2: Scalar float exclusion radius in metres.

    Returns:
        Scalar float32 CBF value; positive is safe, negative is violation.
    """
    return jnp.sum((drone_xy - worker_xy) ** 2) - R2 ** 2


@jax.jit
def h_human_vertical(drone_z, worker_z, H2=H2_EXCLUSION):
    """Vertical CBF for human exclusion cylinder top.

    Args:
        drone_z: Scalar float32 drone altitude in metres.
        worker_z: Scalar float32 worker deck altitude in metres.
        H2: Scalar float exclusion cylinder height in metres.

    Returns:
        Scalar float32 vertical CBF value.
    """
    return drone_z - (worker_z + H2)


@jax.jit
def h_human_predictive(drone_pos, drone_vel, worker_pos, worker_vel, R2=R2_EXCLUSION, H2=H2_EXCLUSION, tau=8, dt=DT):
    """Predictive CBF: min over horizon of radial CBF.

    Args:
        drone_pos: (3,) float32 drone position in metres.
        drone_vel: (3,) float32 drone velocity in metres/second.
        worker_pos: (3,) float32 worker position in metres.
        worker_vel: (3,) float32 worker velocity in metres/second.
        R2: Scalar float exclusion radius in metres.
        H2: Scalar float exclusion height in metres.
        tau: Static int number of prediction steps.
        dt: Scalar float step time in seconds.

    Returns:
        Scalar float32 minimum radial CBF over the horizon.
    """
    def scan_fn(carry, _):
        drone_p, worker_p = carry
        drone_p_next = drone_p + drone_vel * dt
        worker_p_next = worker_p + worker_vel * dt
        h_rad = jnp.sum((drone_p_next[:2] - worker_p_next[:2]) ** 2) - R2 ** 2
        return (drone_p_next, worker_p_next), h_rad
    _, h_vals = jax.lax.scan(scan_fn, (drone_pos, worker_pos), jnp.arange(tau))
    return jnp.min(h_vals)


@jax.jit
def h_structural_pylon(drone_pos, pylon_center_xy, pylon_radius=PYLON_RADIUS, buffer=1.0):
    """CBF for pylon collision avoidance.

    Args:
        drone_pos: (3,) float32 drone position in metres.
        pylon_center_xy: (2,) float32 pylon centre in metres.
        pylon_radius: Scalar float pylon radius in metres.
        buffer: Scalar float clearance buffer in metres.

    Returns:
        Scalar float32 CBF value; positive is safe.
    """
    return jnp.sum((drone_pos[:2] - pylon_center_xy) ** 2) - (pylon_radius + buffer) ** 2


@jax.jit
def h_structural_cable_zone(drone_pos, pylon_x, bridge_width):
    """Signed distance CBF for simplified cable no-fly box.

    Args:
        drone_pos: (3,) float32 drone position in metres.
        pylon_x: Scalar float pylon x position in metres.
        bridge_width: Scalar float bridge width in metres.

    Returns:
        Scalar float32; positive means outside the simplified box.
    """
    dx_lo = drone_pos[0] - (pylon_x - 1.5)
    dx_hi = (pylon_x + 1.5) - drone_pos[0]
    dz_lo = drone_pos[2] - 8.0
    dz_hi = 55.0 - drone_pos[2]
    return jnp.maximum(jnp.maximum(dx_lo, dx_hi), jnp.maximum(dz_lo, dz_hi))


def compute_all_cbf_values(env_state, n_drones, max_workers=MAX_WORKERS):
    """Compute all CBF constraint values for logging/metrics.

    Args:
        env_state: Bridge EnvState with drone/worker/pylon arrays.
        n_drones: Python int number of drones.
        max_workers: Python int padded worker count.

    Returns:
        Dict of float32 arrays: human_radial (n_drones, max_workers),
        human_predictive (n_drones, max_workers), human_vertical
        (n_drones, max_workers), inter_drone (n_drones, n_drones), pylon_1
        (n_drones,), pylon_2 (n_drones,), cable_1 (n_drones,), cable_2
        (n_drones,). Positive values are safe.
    """
    radial = jax.vmap(lambda d: jax.vmap(lambda w: h_human_radial(d[:2], w[:2]))(env_state.worker_pos))(env_state.drone_pos)
    vertical = jax.vmap(lambda d: jax.vmap(lambda w: h_human_vertical(d[2], w[2]))(env_state.worker_pos))(env_state.drone_pos)
    predictive = jax.vmap(lambda dp, dv: jax.vmap(lambda wp, wv: h_human_predictive(dp, dv, wp, wv))(env_state.worker_pos, env_state.worker_vel))(env_state.drone_pos, env_state.drone_vel)
    diff = env_state.drone_pos[:, None, :] - env_state.drone_pos[None, :, :]
    inter = jnp.sum(diff ** 2, axis=-1) - R_SAFE_DRONE ** 2
    pylon_1 = jax.vmap(lambda d: h_structural_pylon(d, env_state.pylon_xy[0]))(env_state.drone_pos)
    pylon_2 = jax.vmap(lambda d: h_structural_pylon(d, env_state.pylon_xy[1]))(env_state.drone_pos)
    cable_1 = jax.vmap(lambda d: h_structural_cable_zone(d, env_state.pylon_xy[0, 0], env_state.bridge_width))(env_state.drone_pos)
    cable_2 = jax.vmap(lambda d: h_structural_cable_zone(d, env_state.pylon_xy[1, 0], env_state.bridge_width))(env_state.drone_pos)
    active = env_state.worker_active[None, :]
    safe_pad = jnp.where(active, 0.0, 1e6)
    return {"human_radial": radial + safe_pad, "human_predictive": predictive + safe_pad, "human_vertical": vertical + safe_pad, "inter_drone": inter, "pylon_1": pylon_1, "pylon_2": pylon_2, "cable_1": cable_1, "cable_2": cable_2}
