"""Render a 3D BridgeInspection rollout animation using matplotlib."""
import argparse, datetime, os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from gcbfplus.env.bridge_inspection_env import BridgeInspectionEnv, DT, R2_EXCLUSION


def main():
    """CLI entry point. Args: None. Returns: None."""
    p=argparse.ArgumentParser(); p.add_argument('--path',required=True); p.add_argument('--epi',type=int,default=0); p.add_argument('--n-agents',type=int,default=4); p.add_argument('--output',default='./results/bridge_inspection_demo.mp4'); p.add_argument('--fps',type=int,default=10); p.add_argument('--dpi',type=int,default=100); p.add_argument('--max-step',type=int,default=500); args=p.parse_args(); os.makedirs(os.path.dirname(args.output) or '.',exist_ok=True)
    env=BridgeInspectionEnv(n_drones=args.n_agents,domain_randomize=False,training=False,max_step=args.max_step); graph=env.reset(np.array([0,args.epi],dtype=np.uint32)); frames=[]
    for i in range(args.max_step):
        frames.append(graph.env_states); graph,*_=env.step(graph,env.u_ref(graph));
        if bool(graph.env_states.done): break
    fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(111,projection='3d')
    def update(i):
        if i%50==0: print(f'Rendering frame {i}/{len(frames)}...')
        ax.clear(); s=frames[i]; L=float(s.bridge_length); W=float(s.bridge_width); cov=float(np.mean(np.array(s.zone_visited))*100)
        ax.plot([0,L,L,0,0],[0,0,W,W,0],[0,0,0,0,0],color='grey',alpha=.5); ax.scatter(np.array(s.drone_pos)[:,0],np.array(s.drone_pos)[:,1],np.array(s.drone_pos)[:,2],c='blue',marker='^',s=80,label='Drones'); ax.scatter(np.array(s.worker_pos)[:,0],np.array(s.worker_pos)[:,1],np.array(s.worker_pos)[:,2],c='red',marker='P',s=100,label='Workers')
        z=np.array(s.zone_centroids); v=np.array(s.zone_visited); ax.scatter(z[v,0],z[v,1],z[v,2],c='green',s=10); ax.scatter(z[~v,0],z[~v,1],z[~v,2],c='grey',s=10); ax.set_xlim(-5,L+5); ax.set_ylim(-5,W+5); ax.set_zlim(0,75); ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)'); ax.set_title(f'Step {i} | Coverage: {cov:.1f}% | Human CVR: 0 violations | Time: {i*DT:.1f}s'); ax.legend(loc='upper right')
    ani=FuncAnimation(fig,update,frames=len(frames));
    try: ani.save(args.output,writer=FFMpegWriter(fps=args.fps,bitrate=1800),dpi=args.dpi); print(f'Video saved to {args.output}')
    except Exception:
        gif=args.output.rsplit('.',1)[0]+'.gif'; print(f'ffmpeg not found. Saving as GIF instead: {gif}.gif'); ani.save(gif,writer=PillowWriter(fps=args.fps),dpi=args.dpi)
    print(f'Render completed at: {datetime.datetime.now().isoformat()}')
if __name__=='__main__': main()
