"""Evaluate BridgeInspection policies and compute metrics from simulation state."""

# Extends GCBF+ (MIT-REALM/gcbfplus), MIT License.
# Original authors: Zhang, So, Garg, Fan (IEEE T-RO 2025).
# This file is an extension; original algorithm is unmodified.

import argparse, datetime, json, os
import jax, jax.numpy as jnp
from gcbfplus.env.bridge_inspection_env import BridgeInspectionEnv, N_ZONES, R2_EXCLUSION, R_SAFE_DRONE, DT


def episode_metrics(env, seed):
    """Run one Python-loop episode.

    Args: env: BridgeInspectionEnv; seed: Python int.
    Returns: Dict of scalar metrics and histories.
    """
    graph = env.reset(jax.random.PRNGKey(seed)); cov_hist=[]; dist_hist=[]; human=0; inter=0; completion=None
    for step in range(env.max_episode_steps):
        s=graph.env_states; action=env.u_ref(graph); graph, *_ = env.step(graph, action, get_eval_info=True); s=graph.env_states
        cov=float(jnp.sum(s.zone_visited)/N_ZONES*100); cov_hist.append(cov)
        d=jnp.linalg.norm(s.drone_pos[:,None,:2]-s.worker_pos[None,:,:2],axis=-1)-R2_EXCLUSION
        d=jnp.where(s.worker_active[None,:], d, 1e6); md=float(jnp.min(d)); dist_hist.append(md)
        human += bool(jnp.any(d < 0)); dd=jnp.linalg.norm(s.drone_pos[:,None,:]-s.drone_pos[None,:,:],axis=-1)+jnp.eye(env.num_agents)*1e6
        inter += bool(jnp.any(dd < R_SAFE_DRONE))
        if completion is None and cov >= 90.0: completion=step+1
        if bool(s.done): break
    total=len(cov_hist); coverage=cov_hist[-1] if cov_hist else 0.0; hc=human/total*100; ic=inter/total*100
    return {"coverage_rate":coverage,"human_cvr":hc,"safety_margin_m":sum(dist_hist)/len(dist_hist),"inter_drone_cvr":ic,"completion_step":completion,"completion_time_s":None if completion is None else completion*DT,"sct_composite":coverage/100*(1-hc/100)*(1-ic/100),"coverage_history":cov_hist,"min_human_dist_history":dist_hist}


def aggregate(episodes):
    """Aggregate episode metrics.

    Args: episodes: List of dict metrics.
    Returns: Dict mean/std for scalar metrics.
    """
    keys=["coverage_rate","human_cvr","safety_margin_m","inter_drone_cvr","sct_composite"]
    return {k:{"mean":float(jnp.mean(jnp.array([e[k] for e in episodes]))),"std":float(jnp.std(jnp.array([e[k] for e in episodes])))} for k in keys}


def main():
    """CLI entry point.

    Args: None. Returns: None.
    """
    p=argparse.ArgumentParser(); p.add_argument('--path',required=True); p.add_argument('--epi',type=int,default=50); p.add_argument('--n-agents',type=int,default=4); p.add_argument('--seed',type=int,default=0); p.add_argument('--baseline',action='store_true'); p.add_argument('--no-video',action='store_true'); p.add_argument('--sensitivity',action='store_true'); args=p.parse_args()
    ts=datetime.datetime.now().isoformat(); env=BridgeInspectionEnv(n_drones=args.n_agents,domain_randomize=False,training=False)
    episodes=[]
    for i in range(args.epi):
        m=episode_metrics(env,args.seed+i); m['episode_id']=i; episodes.append(m)
    agg=aggregate(episodes); comp=[e['completion_time_s'] for e in episodes if e['completion_time_s'] is not None]
    comp_mean=sum(comp)/len(comp) if comp else 0.0
    print('┌─────────────────────────┬────────────┐'); print('│ Metric                  │ Value      │'); print('├─────────────────────────┼────────────┤')
    print(f"│ Coverage Rate (%)       │ {agg['coverage_rate']['mean']:.1f}%      │"); print(f"│ Human CVR (%)           │ {agg['human_cvr']['mean']:.1f}%       │"); print(f"│ Safety Margin (m)       │ {agg['safety_margin_m']['mean']:.2f}m      │"); print(f"│ Inter-Drone CVR (%)     │ {agg['inter_drone_cvr']['mean']:.1f}%       │"); print(f"│ Completion Time (s)     │ {comp_mean:.0f}s       │"); print(f"│ SCT (composite)         │ {agg['sct_composite']['mean']:.2f}       │"); print('└─────────────────────────┴────────────┘')
    os.makedirs('results',exist_ok=True); out=f"results/bridge_eval_{ts.replace(':','')}.json"; json.dump({"run_timestamp":ts,"checkpoint_path":args.path,"n_episodes":args.epi,"n_agents":args.n_agents,"episodes":episodes,"aggregate":agg},open(out,'w'),indent=2); print(out)

if __name__=='__main__': main()
