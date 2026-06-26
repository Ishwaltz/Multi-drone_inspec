"""Tests for the BridgeInspection GCBF+ extension."""

import jax
import jax.numpy as jnp

from gcbfplus.env.bridge_inspection_env import BridgeInspectionEnv, N_ZONES, MAX_WORKERS, DECK_Z, DRONE_MAX_ACCEL
from gcbfplus.env.bridge_cbf import h_human_radial, compute_all_cbf_values


def test_reset_shapes():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(0)); state = graph.env_states
    assert state.drone_pos.shape == (4, 3)
    assert state.zone_visited.shape == (N_ZONES,)
    assert not jnp.any(state.zone_visited)
    assert state.worker_pos.shape == (MAX_WORKERS, 3)


def test_step_jit_compilable():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(0))
    jax.jit(env.step)(graph, jnp.zeros((4, 3)))


def test_obs_shape_n4():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(0))
    assert graph.type_nodes(type_idx=0, n_type=4).shape == (4, 112)


def test_human_cbf_positive_outside():
    assert h_human_radial(jnp.array([10.0, 10.0]), jnp.array([20.0, 20.0])) > 0


def test_human_cbf_negative_inside():
    assert h_human_radial(jnp.array([10.0, 10.0]), jnp.array([10.5, 10.0])) < 0


def test_zone_visited_updates():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(0)); state = graph.env_states
    state = state._replace(drone_pos=state.drone_pos.at[0].set(state.zone_centroids[0]))
    new_graph, *_ = env.step(env.get_graph(state), jnp.zeros((4, 3)))
    assert new_graph.env_states.zone_visited[0]


def test_worker_z_clamped():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(0))
    for _ in range(100):
        graph, *_ = env.step(graph, jnp.zeros((4, 3)))
    state = graph.env_states
    assert jnp.allclose(state.worker_pos[:, 2][state.worker_active], DECK_Z, atol=0.01)


def test_cbf_values_shape():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    state = env.reset(jax.random.PRNGKey(0)).env_states
    vals = compute_all_cbf_values(state, n_drones=4)
    assert vals['human_radial'].shape == (4, MAX_WORKERS)
    assert vals['inter_drone'].shape == (4, 4)
    assert vals['pylon_1'].shape == (4,)


def test_no_nan_100_steps():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=False, training=False)
    graph = env.reset(jax.random.PRNGKey(42))
    for i in range(100):
        action = jax.random.uniform(jax.random.PRNGKey(i), (4, 3), minval=-DRONE_MAX_ACCEL, maxval=DRONE_MAX_ACCEL)
        graph, reward, *_ = env.step(graph, action)
        assert not jnp.any(jnp.isnan(graph.nodes))
        assert not jnp.isnan(reward)
        if graph.env_states.done:
            break


def test_domain_randomization_varies():
    env = BridgeInspectionEnv(n_drones=4, domain_randomize=True, training=False)
    lengths = [float(env.reset(jax.random.PRNGKey(seed)).env_states.bridge_length) for seed in range(10)]
    assert len(set(lengths)) > 1
