"""Generate BridgeInspection proposal plots from JSON result files."""
import argparse, datetime, json, os
import matplotlib.pyplot as plt


def load(path):
    """Load one results JSON. Args: path str. Returns: dict."""
    return json.load(open(path))


def main():
    """CLI entry point. Args: None. Returns: None."""
    p=argparse.ArgumentParser(); p.add_argument('--results-dir',default='results'); p.add_argument('--baselines',nargs='*',default=[]); p.add_argument('--output-dir',default='results/plots'); p.add_argument('--tau-sweep-results'); p.add_argument('--scalability-results'); args=p.parse_args(); os.makedirs(args.output_dir,exist_ok=True); ts=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    labels=['B1: No Safety','B2: Penalty','B3: Static CBF','B5: RRT*','B6: GCBF+ (Predictive)']
    if args.baselines:
        data=[load(x) for x in args.baselines]; h=[d['aggregate']['human_cvr']['mean'] for d in data]; inter=[d['aggregate']['inter_drone_cvr']['mean'] for d in data]
        x=range(len(data)); plt.figure(figsize=(8,5)); plt.bar([i-.2 for i in x],h,.4,hatch='//',label='Human CVR'); plt.bar([i+.2 for i in x],inter,.4,label='Inter-Drone CVR'); plt.xticks(list(x),labels[:len(data)],rotation=20); plt.ylabel('Constraint Violation Rate (%)'); plt.legend(); plt.tight_layout(); out=f'{args.output_dir}/plot1_cvr_baselines_{ts}.png'; plt.savefig(out,dpi=150); print(out); plt.close()
        plt.figure(figsize=(8,5)); plt.axvspan(0,2,color='green',alpha=.15,label='Safe Operating Zone');
        for lab,d in zip(labels,data): plt.scatter([e['human_cvr'] for e in d['episodes']],[e['coverage_rate'] for e in d['episodes']],label=lab)
        plt.xlabel('Human CVR (%)'); plt.ylabel('Coverage Rate (%)'); plt.legend(fontsize=8); plt.tight_layout(); out=f'{args.output_dir}/plot2_safety_efficiency_{ts}.png'; plt.savefig(out,dpi=150); print(out); plt.close()
    if args.tau_sweep_results: print(f"Plot 3 input: {args.tau_sweep_results}")
    else: print('SKIPPED: tau sweep results not provided. Run eval with --tau-sweep flag.')
    if args.scalability_results: print(f"Plot 4 input: {args.scalability_results}")
    else: print('SKIPPED: scalability results not provided.')
if __name__=='__main__': main()
