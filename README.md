<div align="center">

# Multi-Drone Human-Aware Bridge Inspection System

A human-aware multi-drone autonomous inspection framework for safe bridge inspection operations with active workers on-site.

</div>

## Overview

This project extends the [GCBF+](https://mit-realm.github.io/gcbfplus-website/) neural graph control barrier function framework to enable safe autonomous multi-drone bridge inspection in human-populated environments. By integrating human worker behavioral models and proximity-based safety constraints, this system enables drones to complete inspection tasks while maintaining safety guarantees around human workers.

### Key Features

- **Multi-Drone Coordination**: Leverages GCBF+ anti-collision capabilities for coordinated drone operations
- **Human Worker Models**: Simulates realistic worker behaviors including:
  - Standing still at work stations
  - Crouching/bending positions
  - Moving between inspection stations
  - Dynamic positional changes
- **Human-Aware Safety Boundaries**: Proximity-based safety constraints with graduated penalties:
  - Distance-dependent penalty scaling to encourage drones to maintain safe standoff distances
  - Behavior-aware constraints that adapt based on worker pose and movement
  - Configurable safety margins for different inspection scenarios
- **Autonomous Bridge Inspection**: Drones execute inspection trajectories while respecting both anti-collision and human safety constraints

<div align="center">
    <img src="./media/cbf1.gif" alt="LidarSpread" width="24.55%"/>
    <img src="./media/DoubleIntegrator_512_2x.gif" alt="LidarLine" width="24.55%"/>
    <img src="./media/Obstacle2D_32.gif" alt="VMASReverseTransport" width="24.55%"/>
    <img src="./media/Obstacle2D_512_2x.gif" alt="VMASWheel" width="24.55%"/>
</div>

## Dependencies

We recommend using [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n gcbfplus python=3.10
conda activate gcbfplus
cd Multi-drone_inspec
```

Then install JAX following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install the package:

```bash
pip install -e .
```

## Run

### Environments

We provide 3 2D environments including `SingleIntegrator`, `DoubleIntegrator`, and `DubinsCar`, and 2 3D environments including `LinearDrone` and `CrazyFlie`.

**New in this fork**: Human worker models can be instantiated in any environment to simulate realistic inspection scenarios with active workers on-site.

### Algorithms

We provide algorithms including GCBF+ (`gcbf+`), GCBF (`gcbf`), centralized CBF-QP (`centralized_cbf`), and decentralized CBF-QP (`dec_share_cbf`). Use `--algo` to specify the algorithm. 

### Human Worker Configuration

Human workers are configured with:
- **Behavior modes**: `standing`, `crouching`, `moving_between_stations`, `standing_still`
- **Safety boundary radius**: Configurable distance threshold around each worker
- **Penalty scaling**: Distance-based penalty weights that increase as drones approach the worker

### Hyper-parameters

To reproduce the results shown in the original GCBF+ paper, refer to [`settings.yaml`](./settings.yaml). For human-aware configurations, see the human safety parameters section.

### Train

To train a human-aware multi-drone inspection model, use:

```bash
python train.py --algo gcbf+ --env DoubleIntegrator -n 8 --area-size 4 --loss-action-coef 1e-4 --n-env-train 16 --lr-actor 1e-5 --lr-cbf 1e-5 --horizon 32 --human-workers 2 --human-safety-penalty 0.5
```

Key parameters for human-aware training:
- `--human-workers`: number of human workers in the environment
- `--human-safety-penalty`: base penalty weight for proximity to humans (higher = stricter safety constraints)
- `--human-safety-radius`: minimum safe distance from human workers
- `--human-behavior-modes`: worker behavior distribution during training

Standard training flags:
- `-n`: number of drones
- `--env`: environment
- `--algo`: algorithm (`gcbf+` or `gcbf` for learning-based approaches)
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: experiment name
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of training environments
- `--n-env-test`: number of test environments
- `--log-dir`: path to save logs
- `--eval-interval`: evaluation frequency
- `--eval-epi`: episodes per evaluation
- `--save-interval`: checkpoint save frequency

Hyperparameter flags:
- `--alpha`: GCBF alpha
- `--horizon`: GCBF+ lookahead horizon
- `--lr-actor`: actor learning rate
- `--lr-cbf`: CBF learning rate
- `--loss-action-coef`: action loss weight
- `--loss-h-dot-coef`: h_dot loss weight
- `--loss-safe-coef`: safety loss weight
- `--loss-unsafe-coef`: unsafe state loss weight
- `--buffer-size`: replay buffer size

### Test

To test the human-aware inspection model:

```bash
python test.py --path <path-to-log> --epi 5 --area-size 4 -n 8 --human-workers 2 --human-behavior-distribution varied
```

This evaluates:
- Safety rate (collision avoidance with drones and obstacles)
- Human safety rate (proximity violations with workers)
- Goal reaching rate (inspection task completion)
- Success rate (safe inspection completion)

Generated videos will be saved in `<path-to-log>/videos`.

Test flags:
- `-n`: number of drones
- `--human-workers`: number of human workers
- `--human-behavior-distribution`: worker behavior profile (`static`, `dynamic`, `varied`)
- `--human-safety-radius`: safety boundary radius
- `--area-size`: environment size
- `--max-step`: maximum episode length
- `--path`: log folder path
- `--epi`: number of test episodes
- `--seed`: random seed
- `--no-video`: skip video generation
- `--log`: save results to file

Example: Test with static workers (standing still):
```bash
python test.py --env DoubleIntegrator -n 8 --human-workers 3 --human-behavior-distribution static --epi 5 --area-size 4
```

Example: Test with dynamic workers (moving between stations):
```bash
python test.py --env DoubleIntegrator -n 8 --human-workers 3 --human-behavior-distribution dynamic --epi 5 --area-size 4
```

### Pre-trained Models

Pre-trained models are available in the [`pretrained`](pretrained) folder. However, performance may vary depending on GPU/CUDA/JAX versions. **Retraining on your specific hardware is recommended**, especially for human-aware scenarios where safety is critical.

## Citation

If you use this work, please cite the original GCBF+ paper:

```
@ARTICLE{zhang2025gcbf+,
      author={Zhang, Songyuan and So, Oswin and Garg, Kunal and Fan, Chuchu},
      journal={IEEE Transactions on Robotics}, 
      title={{GCBF}+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control}, 
      year={2025},
      volume={41},
      pages={1533-1552},
      doi={10.1109/TRO.2025.3530348}
}
```

For this human-aware extension, please reference this repository:
```
@misc{ishwaltz2024multidroneinspection,
      title={Multi-Drone Human-Aware Bridge Inspection System},
      author={Ishwaltz},
      year={2024},
      howpublished={\url{https://github.com/Ishwaltz/Multi-drone_inspec}}
}
```

## Acknowledgements

This work builds upon the [GCBF+](https://mit-realm.github.io/gcbfplus-website/) framework developed by Songyuan Zhang, Oswin So, Kunal Garg, and Chuchu Fan at MIT.

The original developers were partially supported by MITRE during the GCBF+ project.

© 2024 MIT  
© 2024 The MITRE Corporation  
© 2024 Human-Aware Extension Contributors
