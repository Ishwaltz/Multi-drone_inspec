"""BridgeInspection environment extending GCBF+ under the MIT license."""

import functools as ft
import pathlib
from typing import Tuple, Optional, Dict, Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from gcbfplus.env.base import MultiAgentEnv, RolloutResult
from gcbfplus.utils.graph import EdgeBlock, GetGraph, GraphsTuple
from gcbfplus.utils.typing import Action, Array, Cost, Done, Info, Reward, State
from gcbfplus.env import bridge_geometry
from gcbfplus.env.bridge_cbf import compute_all_cbf_values

BRIDGE_LENGTH_DEFAULT = 200.0  # 200m main span; Gimsing & Georgakis (2012), Cable Supported Bridges.
BRIDGE_WIDTH_DEFAULT = 20.0  # 20m four-lane deck; IRC:6-2017 carriageway widths.
DECK_Z = 0.0  # Deck reference z-level; all heights are relative to deck.
PYLON_HEIGHT = 80.0  # 40% of 200m span; Walther et al. (1999), Cable Stayed Bridges.
PYLON_RADIUS = 2.0  # 2m pylon radius; Rion-Antirion bridge-class structural drawings.
PYLON_XY_DEFAULT = [(50.0, 10.0), (150.0, 10.0)]  # 25%/75% span, centred on 20m deck; Gimsing & Georgakis (2012).
BRIDGE_LENGTH_RANGE = (100.0, 250.0)  # metres; specified domain randomisation range.
BRIDGE_WIDTH_RANGE = (15.0, 30.0)  # metres; specified domain randomisation range.
N_WORKERS_RANGE = (1, 5)  # inclusive; specified domain randomisation range.
WIND_MAG_RANGE = (0.0, 3.0)  # m/s additive velocity noise; specified domain randomisation range.
N_DRONES_DEFAULT = 4  # default team size specified for balanced workload.
DRONE_COMM_RADIUS = 10.0  # 5 × r_safe_drone; extends GCBF+ r_comm=4×r_safe for 3D separation.
R_DRONE_BODY = 0.5  # 1m diameter class; DJI Matrice 350 RTK wheelbase reference.
R_SAFE_DRONE = 2.0  # 4 × body radius for downwash clearance; Shukla & Bhatt (2018).
DRONE_MAX_ACCEL = 5.0  # m/s² conservative inspection-drone acceleration; Mahony et al. (2012).
DRONE_MAX_VEL = 8.0  # m/s Category 3 RPAS limit; DGCA India (2021).
DRONE_Z_MIN = 2.5  # H2 2.3m + 0.2m buffer to avoid worker cylinder/deck strike.
DRONE_Z_MAX = 70.0  # 0.875 × 80m pylon height; above this no inspection value.
DRONE_Z_INIT_RANGE = (5.0, 10.0)  # metres above deck; specified safe initial altitude.
DRONE_MIN_INIT_SPACING = 5.0  # metres; specified reset rejection spacing.
SENSOR_NOISE_STD = 0.05  # sim-to-real observation randomisation; Tobin et al. (2017), OpenAI.
N_WORKERS_DEFAULT = 3  # 3-person crew; FHWA BIRM (2012) typical 2–4 persons.
R1_STANDING = 0.45  # m body cylinder with equipment allowance; NASA Anthropometric Source Book.
R1_TOOL_USE = 0.80  # m comfortable reach; OSHA ergonomic reach reference.
R2_EXCLUSION = 1.5  # m hard exclusion radius; ISO 13855:2010 protective separation derivation.
H1_WORKER = 1.75  # m worker height; Pheasant (2003) 95th percentile male standing height.
H2_EXCLUSION = 2.3  # m H1 + 0.55m overhead buffer; OSHA overhead reach reference.
MAX_WORKERS = 5  # JAX fixed-shape padding dimension specified.
N_ZONES = 48  # 16 deck + 16 cable + 16 pylon zones; specified balanced workload.
ZONE_VISIT_RADIUS = 2.5  # m camera footprint estimate at 5m altitude and 90-degree FoV.
N_CABLE_ZONES_PER_PYLON = 8  # specified cable zones per pylon.
CABLE_ZONE_HEIGHTS = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]  # m lower stay attachment range.
CABLE_ZONE_OFFSET = 3.0  # m horizontal offset from pylon centre; specified simplification.
N_PYLON_ZONES_PER_PYLON = 8  # specified pylon zones per pylon.
PYLON_INSPECT_OFFSET = 3.5  # PYLON_RADIUS 2.0m + 1.5m standoff.
W_COV = 1.0  # coverage reward weight specified.
W_TASK = 5.0  # task completion bonus weight specified.
W_PROX = 2.0  # human proximity penalty weight specified.
W_INTER = 3.0  # inter-drone collision penalty weight specified.
W_FORM = 0.5  # formation looseness penalty weight specified.
W_ENERGY = 0.05  # action energy penalty weight specified.
TARGET_DRONE_SPACING = 8.0  # bridge_width/(N-1) ≈ 8m for W=20, N=4.
FRONT_APPROACH_MULTIPLIER = 1.5  # ISO 11228-1:2003 direction-risk inspired multiplier.
MAX_STEPS = 2000  # 200 seconds at DT=0.1s; specified episode length.
DT = 0.1  # seconds; specified timestep.


class BridgeInspectionEnv(MultiAgentEnv):
    """Cable-stayed bridge inspection environment with drones and worker nodes."""

    AGENT = 0
    WORKER = 1
    PARAMS = {"comm_radius": DRONE_COMM_RADIUS, "drone_radius": R_DRONE_BODY}

    class EnvState(NamedTuple):
        drone_pos: Array
        drone_vel: Array
        worker_pos: Array
        worker_vel: Array
        worker_state_id: Array
        worker_goal: Array
        worker_speed: Array
        worker_t_elapsed: Array
        worker_active: Array
        zone_visited: Array
        zone_centroids: Array
        n_workers: Array
        bridge_length: Array
        bridge_width: Array
        pylon_xy: Array
        wind_vel: Array
        step_count: Array
        done: Array
        key: Array

        @property
        def n_agent(self) -> int:
            return self.drone_pos.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    def __init__(self, num_agents: int = N_DRONES_DEFAULT, area_size: float = BRIDGE_LENGTH_DEFAULT, max_step: int = MAX_STEPS, max_travel: float = None, dt: float = DT, params: dict = None, domain_randomize: bool = True, training: bool = True, n_drones: Optional[int] = None):
        """Initialise the bridge inspection environment.

        Args:
            num_agents: Python int number of drones.
            area_size: Scalar float retained for GCBF+ factory compatibility.
            max_step: Python int maximum rollout steps.
            max_travel: Optional scalar retained for base compatibility.
            dt: Scalar float integration timestep in seconds.
            params: Optional dict of environment parameters.
            domain_randomize: Bool enabling reset-time geometry randomisation.
            training: Bool enabling observation noise.
            n_drones: Optional alias for num_agents used by tests/scripts.

        Returns:
            None.
        """
        if n_drones is not None:
            num_agents = n_drones
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params or self.PARAMS.copy())
        self.domain_randomize = domain_randomize
        self.training = training

    @property
    def state_dim(self) -> int: return 6
    @property
    def node_dim(self) -> int: return 112
    @property
    def edge_dim(self) -> int: return 6
    @property
    def action_dim(self) -> int: return 3

    def _sample_reset_values(self, key: Array):
        """Sample reset geometry values using JAX keys.

        Args:
            key: (2,) uint32 JAX PRNG key.

        Returns:
            Tuple of bridge_length, bridge_width, n_workers, wind_vel, key.
        """
        if self.domain_randomize:
            k1, k2, k3, k4, key = jr.split(key, 5)
            length = jr.uniform(k1, (), minval=BRIDGE_LENGTH_RANGE[0], maxval=BRIDGE_LENGTH_RANGE[1])
            width = jr.uniform(k2, (), minval=BRIDGE_WIDTH_RANGE[0], maxval=BRIDGE_WIDTH_RANGE[1])
            n_workers = jr.randint(k3, (), minval=N_WORKERS_RANGE[0], maxval=N_WORKERS_RANGE[1] + 1)
            wind_xy = jr.uniform(k4, (2,), minval=-WIND_MAG_RANGE[1], maxval=WIND_MAG_RANGE[1])
            wind = jnp.array([wind_xy[0], wind_xy[1], 0.0], dtype=jnp.float32)
        else:
            length = jnp.array(BRIDGE_LENGTH_DEFAULT, dtype=jnp.float32)
            width = jnp.array(BRIDGE_WIDTH_DEFAULT, dtype=jnp.float32)
            n_workers = jnp.array(N_WORKERS_DEFAULT, dtype=jnp.int32)
            wind = jnp.zeros(3, dtype=jnp.float32)
        return length, width, n_workers, wind, key

    def reset(self, key: Array) -> GraphsTuple:
        """Reset and return a padded graph.

        Args:
            key: (2,) uint32 JAX PRNG key.

        Returns:
            GraphsTuple with Bridge EnvState in graph.env_states.
        """
        self._t = 0
        length, width, n_workers, wind, key = self._sample_reset_values(key)
        length_f, width_f, nw_i = float(length), float(width), int(n_workers)
        pylon_xy_np = np.array([[0.25 * length_f, 0.5 * width_f], [0.75 * length_f, 0.5 * width_f]], dtype=np.float32)
        zones_np = bridge_geometry.get_zone_centroids(length_f, width_f, pylon_xy_np)
        seed = int(jax.device_get(key[0]))
        workers_np = bridge_geometry.get_initial_worker_positions(nw_i, zones_np, seed)
        rng = np.random.default_rng(seed + 17)
        drone_pos = None
        for _ in range(100):
            cand = np.column_stack([
                rng.uniform(0.0, length_f, size=self.num_agents),
                rng.uniform(0.0, width_f, size=self.num_agents),
                rng.uniform(DRONE_Z_INIT_RANGE[0], DRONE_Z_INIT_RANGE[1], size=self.num_agents),
            ]).astype(np.float32)
            dist = np.linalg.norm(cand[:, None, :] - cand[None, :, :], axis=-1) + np.eye(self.num_agents) * 1e6
            if np.all(dist >= DRONE_MIN_INIT_SPACING):
                drone_pos = cand
                break
        if drone_pos is None:
            xs = np.linspace(0.2 * length_f, 0.8 * length_f, self.num_agents, dtype=np.float32)
            ys = np.full(self.num_agents, 0.5 * width_f, dtype=np.float32)
            zs = np.full(self.num_agents, DRONE_Z_INIT_RANGE[0], dtype=np.float32)
            drone_pos = np.stack([xs, ys, zs], axis=1)
        worker_pos = np.zeros((MAX_WORKERS, 3), dtype=np.float32)
        worker_pos[:nw_i] = workers_np
        active = np.zeros((MAX_WORKERS,), dtype=bool); active[:nw_i] = True
        state = self.EnvState(
            jnp.asarray(drone_pos, dtype=jnp.float32), jnp.zeros((self.num_agents, 3), dtype=jnp.float32),
            jnp.asarray(worker_pos, dtype=jnp.float32), jnp.zeros((MAX_WORKERS, 3), dtype=jnp.float32),
            jnp.where(jnp.asarray(active), 1, 0).astype(jnp.int32), jnp.asarray(worker_pos, dtype=jnp.float32),
            jnp.ones((MAX_WORKERS,), dtype=jnp.float32), jnp.zeros((MAX_WORKERS,), dtype=jnp.float32),
            jnp.asarray(active), jnp.zeros((N_ZONES,), dtype=bool), jnp.asarray(zones_np, dtype=jnp.float32),
            n_workers, length, width, jnp.asarray(pylon_xy_np, dtype=jnp.float32), wind,
            jnp.array(0, dtype=jnp.int32), jnp.array(False), key)
        return self.get_graph(state)

    def _worker_step(self, state: EnvState):
        """Advance worker Markov dynamics with vectorised JAX operations.

        Args:
            state: Bridge EnvState.

        Returns:
            Tuple of updated worker fields and PRNG key.
        """
        key, kr, kn, kg = jr.split(state.key, 4)
        u = jr.uniform(kr, (MAX_WORKERS,))
        noise = jr.normal(kn, (MAX_WORKERS, 3)) * 0.01
        to_goal = state.worker_goal - state.worker_pos
        dist = jnp.linalg.norm(to_goal[:, :2], axis=-1, keepdims=True)
        direction = to_goal / jnp.maximum(dist, 1e-6)
        transit_pos = state.worker_pos + direction * state.worker_speed[:, None] * self.dt
        transit_pos = transit_pos.at[:, 2].set(DECK_Z)
        station_pos = (state.worker_pos + noise).at[:, 2].set(DECK_Z)
        pos_pre = jnp.where((state.worker_state_id == 0)[:, None], transit_pos, state.worker_pos)
        pos_pre = jnp.where((state.worker_state_id == 1)[:, None], station_pos, pos_pre)
        pos_pre = jnp.where(state.worker_active[:, None], pos_pre, 0.0)
        arrived = jnp.sum((pos_pre[:, :2] - state.worker_goal[:, :2]) ** 2, axis=-1) < 1.0
        # Physical estimate — not empirically validated.
        # See sensitivity analysis in eval/run_bridge_eval.py.
        sid = state.worker_state_id
        new_sid = sid
        new_sid = jnp.where((sid == 0) & arrived, jnp.where(u < 0.8, 1, 3), new_sid)
        new_sid = jnp.where((sid == 1) & (u < 0.002), 0, new_sid)
        new_sid = jnp.where((sid == 1) & (u >= 0.002) & (u < 0.007), 2, new_sid)
        new_sid = jnp.where((sid == 2) & (u < 0.01), 1, new_sid)
        new_sid = jnp.where((sid == 3) & (u < 0.003), 0, new_sid)
        goal_idx = jr.randint(kg, (MAX_WORKERS,), 0, 16)
        new_goals = state.zone_centroids[goal_idx].at[:, 2].set(DECK_Z)
        goal = jnp.where((new_sid == 0)[:, None] & (sid != 0)[:, None], new_goals, state.worker_goal)
        t_elapsed = jnp.where(new_sid == sid, state.worker_t_elapsed + self.dt, 0.0)
        vel = (pos_pre - state.worker_pos) / self.dt
        return pos_pre, vel, new_sid.astype(jnp.int32), goal, state.worker_speed, t_elapsed, key

    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False):
        """Advance one simulation step.

        Args:
            graph: GraphsTuple whose env_states is Bridge EnvState.
            action: (n_drones, 3) float32 acceleration command.
            get_eval_info: Bool; when true include CBF metrics in info.

        Returns:
            Tuple (next_graph, reward, cost, done, info).
        """
        state = graph.env_states if isinstance(graph, GraphsTuple) else graph
        action = self.clip_action(action)
        vel = jnp.clip(state.drone_vel + action * self.dt + state.wind_vel * self.dt, -DRONE_MAX_VEL, DRONE_MAX_VEL)
        pos = state.drone_pos + vel * self.dt
        pos = pos.at[:, 2].set(jnp.clip(pos[:, 2], DRONE_Z_MIN, DRONE_Z_MAX))
        wpos, wvel, wsid, wgoal, wspeed, wtel, key = self._worker_step(state)
        dist_mat = jnp.sqrt(jnp.sum((pos[:, None, :] - state.zone_centroids[None, :, :]) ** 2, axis=-1) + 1e-8)
        newly = jnp.any(dist_mat < ZONE_VISIT_RADIUS, axis=0)
        visited = jnp.logical_or(state.zone_visited, newly)
        hrad = jnp.sum((pos[:, None, :2] - wpos[None, :, :2]) ** 2, axis=-1) - R2_EXCLUSION ** 2
        in_cyl = (hrad < 0) & (pos[:, None, 2] < (wpos[None, :, 2] + H2_EXCLUSION)) & state.worker_active[None, :]
        wf = wgoal - wpos
        wf = wf / jnp.maximum(jnp.linalg.norm(wf, axis=-1, keepdims=True), 1e-6)
        toward = wpos[None, :, :] - pos[:, None, :]
        toward = toward / jnp.maximum(jnp.linalg.norm(toward, axis=-1, keepdims=True), 1e-6)
        front = jnp.where(jnp.sum(toward * wf[None, :, :], axis=-1) > 0.5, FRONT_APPROACH_MULTIPLIER, 1.0)
        prox_pen = W_PROX * jnp.sum(front * in_cyl.astype(jnp.float32))
        dd = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) + jnp.eye(self.num_agents) * 1e6
        inter_pen = W_INTER * jnp.sum(jnp.triu((dd < R_SAFE_DRONE).astype(jnp.float32), 1))
        nearest = jnp.min(dd, axis=1)
        form_pen = W_FORM * jnp.mean(jnp.abs(nearest - TARGET_DRONE_SPACING))
        energy_pen = W_ENERGY * jnp.sum(action ** 2)
        r_cov = W_COV * jnp.sum(newly).astype(jnp.float32) * 0.1
        r_task = jnp.where(jnp.all(visited) & (~jnp.all(state.zone_visited)), W_TASK * 5.0, 0.0)
        reward = r_cov + r_task - prox_pen - inter_pen - form_pen - energy_pen
        done = jnp.all(visited) | jnp.any(pos[:, 0] < -5.0) | jnp.any(pos[:, 0] > state.bridge_length + 5.0) | jnp.any(pos[:, 1] < -5.0) | jnp.any(pos[:, 1] > state.bridge_width + 5.0) | (state.step_count + 1 >= self.max_episode_steps)
        next_state = state._replace(drone_pos=pos, drone_vel=vel, worker_pos=wpos, worker_vel=wvel, worker_state_id=wsid, worker_goal=wgoal, worker_speed=wspeed, worker_t_elapsed=wtel, zone_visited=visited, step_count=state.step_count + 1, done=done, key=key)
        cost = jnp.mean(jnp.any(in_cyl, axis=1).astype(jnp.float32)) + jnp.mean(jnp.any(dd < R_SAFE_DRONE, axis=1).astype(jnp.float32))
        info = compute_all_cbf_values(next_state, self.num_agents) if get_eval_info else {}
        return self.get_graph(next_state), reward.astype(jnp.float32), cost.astype(jnp.float32), done, info

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        """Build a padded radius-neighbour graph.

        Args:
            state: Bridge EnvState.
            adjacency: Optional unused adjacency override.

        Returns:
            GraphsTuple with drone and worker nodes.
        """
        obs = self._obs(state)
        n_nodes = self.num_agents + MAX_WORKERS
        worker_nodes = jnp.zeros((MAX_WORKERS, self.node_dim), dtype=jnp.float32)
        nodes = jnp.concatenate([obs, worker_nodes], axis=0)
        node_type = jnp.concatenate([jnp.zeros((self.num_agents,), dtype=jnp.int32), jnp.ones((MAX_WORKERS,), dtype=jnp.int32)], axis=0)
        drone_state = jnp.concatenate([state.drone_pos, state.drone_vel], axis=1)
        worker_state = jnp.concatenate([state.worker_pos, state.worker_vel], axis=1)
        states = jnp.concatenate([drone_state, worker_state], axis=0)
        pos = state.drone_pos
        diff = pos[:, None, :] - pos[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1) + jnp.eye(self.num_agents) * (DRONE_COMM_RADIUS + 1.0)
        mask = dist < DRONE_COMM_RADIUS
        feats = drone_state[:, None, :] - drone_state[None, :, :]
        aa = EdgeBlock(feats, mask, jnp.arange(self.num_agents), jnp.arange(self.num_agents))
        wdiff = drone_state[:, None, :] - worker_state[None, :, :]
        wm = jnp.broadcast_to(state.worker_active[None, :], (self.num_agents, MAX_WORKERS))
        aw = EdgeBlock(wdiff, wm, jnp.arange(self.num_agents), jnp.arange(self.num_agents, n_nodes))
        return GetGraph(nodes=nodes, node_type=node_type, edge_blocks=[aa, aw], env_states=state, states=states).to_padded()

    def _obs(self, state: EnvState) -> Array:
        """Construct per-drone 112D observations.

        Args:
            state: Bridge EnvState.

        Returns:
            (n_drones, 112) float32 node feature matrix.
        """
        own = jnp.concatenate([state.drone_pos, state.drone_vel], axis=1)
        diff = state.drone_pos[None, :, :] - state.drone_pos[:, None, :]
        dist = jnp.linalg.norm(diff, axis=-1) + jnp.eye(self.num_agents) * 1e6
        in_rad = dist < DRONE_COMM_RADIUS
        order = jnp.argsort(jnp.where(in_rad, dist, 1e6), axis=1)[:, :N_DRONES_DEFAULT - 1]
        rel_pos = jnp.take_along_axis(diff, order[:, :, None], axis=1)
        vel_j = jnp.broadcast_to(state.drone_vel[None, :, :], (self.num_agents, self.num_agents, 3))
        rel_vel = jnp.take_along_axis(vel_j, order[:, :, None], axis=1)
        valid = jnp.take_along_axis(in_rad, order, axis=1)[:, :, None]
        rel_drone = jnp.where(valid, jnp.concatenate([rel_pos, rel_vel], axis=-1), 0.0).reshape((self.num_agents, -1))
        rel_worker_pos = state.worker_pos[None, :, :] - state.drone_pos[:, None, :]
        rel_worker = jnp.concatenate([rel_worker_pos, jnp.broadcast_to(state.worker_state_id[None, :, None].astype(jnp.float32), (self.num_agents, MAX_WORKERS, 1))], axis=-1)
        rel_worker = jnp.where(state.worker_active[None, :, None], rel_worker, 0.0).reshape((self.num_agents, -1))
        zd = state.zone_centroids[None, :, :] - state.drone_pos[:, None, :]
        zdist = jnp.linalg.norm(zd, axis=-1)
        zorder = jnp.argsort(jnp.where(~state.zone_visited[None, :], zdist, 1e6), axis=1)[:, :6]
        zrel = jnp.take_along_axis(zd, zorder[:, :, None], axis=1)
        zvalid = jnp.take_along_axis(~state.zone_visited[None, :].repeat(self.num_agents, axis=0), zorder, axis=1)[:, :, None]
        zrel = jnp.where(zvalid, zrel, 0.0).reshape((self.num_agents, -1))
        bridge = jnp.broadcast_to(jnp.array([state.bridge_length, state.bridge_width], dtype=jnp.float32), (self.num_agents, 2))
        mask = jnp.broadcast_to(state.zone_visited.astype(jnp.float32), (self.num_agents, N_ZONES))
        noisy = jnp.concatenate([own, rel_drone, rel_worker, zrel], axis=1)
        if self.training:
            keys = jr.split(state.key, self.num_agents)
            noise = jax.vmap(lambda k: jr.normal(k, noisy.shape[1:]) * SENSOR_NOISE_STD)(keys)
            noisy = noisy + noise
        return jnp.concatenate([noisy, bridge, mask], axis=1)

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """Return state limits.

        Args:
            state: Optional state ignored.

        Returns:
            Tuple of (6,) float32 lower/upper limits.
        """
        return jnp.array([-jnp.inf, -jnp.inf, DRONE_Z_MIN, -DRONE_MAX_VEL, -DRONE_MAX_VEL, -DRONE_MAX_VEL]), jnp.array([jnp.inf, jnp.inf, DRONE_Z_MAX, DRONE_MAX_VEL, DRONE_MAX_VEL, DRONE_MAX_VEL])

    def action_lim(self) -> Tuple[Action, Action]:
        """Return acceleration limits.

        Args:
            None.

        Returns:
            Tuple of (3,) float32 lower/upper action limits.
        """
        return -jnp.ones(3) * DRONE_MAX_ACCEL, jnp.ones(3) * DRONE_MAX_ACCEL

    def control_affine_dyn(self, state: State):
        """Return double-integrator control-affine dynamics.

        Args:
            state: (n_drones, 6) float32 drone states.

        Returns:
            Tuple f (n_drones, 6), g (n_drones, 6, 3).
        """
        f = jnp.concatenate([state[:, 3:], jnp.zeros_like(state[:, 3:])], axis=1)
        g = jnp.zeros((state.shape[0], 6, 3)).at[:, 3:, :].set(jnp.eye(3))
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        """Update edge features from node states.

        Args:
            graph: Single GraphsTuple.
            state: (n_nodes, 6) float32 states.

        Returns:
            GraphsTuple with updated states/edges.
        """
        return graph._replace(edges=state[graph.receivers] - state[graph.senders], states=state)

    def u_ref(self, graph: GraphsTuple) -> Action:
        """Reference controller drives drones toward nearest unvisited zone.

        Args:
            graph: GraphsTuple.

        Returns:
            (n_drones, 3) float32 acceleration action.
        """
        s = graph.env_states
        zd = s.zone_centroids[None, :, :] - s.drone_pos[:, None, :]
        idx = jnp.argmin(jnp.where(~s.zone_visited[None, :], jnp.linalg.norm(zd, axis=-1), 1e6), axis=1)
        target = s.zone_centroids[idx]
        return self.clip_action((target - s.drone_pos) * 0.5 - s.drone_vel * 0.2)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        """Predict next graph for action.

        Args:
            graph: GraphsTuple.
            action: (n_drones, 3) float32 action.

        Returns:
            Next GraphsTuple.
        """
        return self.step(graph, action)[0]

    def safe_mask(self, graph: GraphsTuple) -> Array:
        """Return per-agent safety mask.

        Args:
            graph: GraphsTuple.

        Returns:
            (n_drones,) bool safety mask.
        """
        return ~self.collision_mask(graph)

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        """Return per-agent unsafe mask.

        Args:
            graph: GraphsTuple.

        Returns:
            (n_drones,) bool unsafe mask.
        """
        return self.collision_mask(graph)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        """Return per-agent collision mask.

        Args:
            graph: GraphsTuple.

        Returns:
            (n_drones,) bool collision mask.
        """
        s = graph.env_states
        dd = jnp.linalg.norm(s.drone_pos[:, None, :] - s.drone_pos[None, :, :], axis=-1) + jnp.eye(self.num_agents) * 1e6
        inter = jnp.any(dd < R_SAFE_DRONE, axis=1)
        h = jnp.sum((s.drone_pos[:, None, :2] - s.worker_pos[None, :, :2]) ** 2, axis=-1) - R2_EXCLUSION ** 2
        human = jnp.any((h < 0) & s.worker_active[None, :] & (s.drone_pos[:, None, 2] < s.worker_pos[None, :, 2] + H2_EXCLUSION), axis=1)
        return inter | human

    def finish_mask(self, graph: GraphsTuple) -> Array:
        """Return per-agent finish mask.

        Args:
            graph: GraphsTuple.

        Returns:
            (n_drones,) bool finish mask.
        """
        return jnp.ones((self.num_agents,), dtype=bool) * jnp.all(graph.env_states.zone_visited)

    def render_video(self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, **kwargs) -> None:
        """Render placeholder delegated to eval/render_video.py.

        Args:
            rollout: RolloutResult trajectory.
            video_path: Output path.
            Ta_is_unsafe: Optional unsafe mask.
            viz_opts: Optional visualisation options.
            **kwargs: Extra options.

        Returns:
            None.
        """
        raise NotImplementedError("Use eval/render_video.py for BridgeInspection videos.")
