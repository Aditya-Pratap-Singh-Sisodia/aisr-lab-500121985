# Experiment 7: Mobile Robot Simulation in Webots: Reactive Swarm and Lidar Navigation

## Platform and Environment
- Simulator: Webots R2025a
- Operating System: Windows 11 Home Single Language (Build 26100, x86_64)
- CPU / Hardware: Intel Core i5-11400H @ 2.70 GHz (12 logical processors)
- Python Version: Python 3.12.10
- Simulator Controllers Modified:
  - `controllers/epuck_braitenberg/`
  - `controllers/pioneer_lidar_nav/`
  - `controllers/trajectory_logger/`

## Simulation Worlds Used
1. `worlds/swarm_arena.wbt`:
   - Robots: 5 × e-puck differential drive robots with 8 infrared proximity sensors (`ps0` to `ps7`)
   - Obstacles: 3 wooden boxes, 2 oil barrels inside a 1.6 m × 1.6 m walled arena
   - Controller: Braitenberg reactive obstacle avoidance
   - Run Duration: 120 s simulated time per run (3 to 5 repeats per configuration)
2. `worlds/pioneer_lidar.wbt`:
   - Robot: Pioneer 3-DX mobile base equipped with SICK LMS 291 planar scanning lidar (180° field of view)
   - Environment: Cluttered 8 m × 8 m indoor room with columns, walls, and narrow doorways
   - Controller: Follow-The-Gap reactive navigation
   - Run Duration: 180 s simulated time per run

## Execution Commands Run
```bash
# Swarm Part A Runs (120 s, 3 repeats per setting)
python tools/run_experiment.py swarm --seconds 120 --name swarm_default_rep1
python tools/run_experiment.py swarm --seconds 120 --name swarm_default_rep2
python tools/run_experiment.py swarm --seconds 120 --name swarm_default_rep3
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.2 --name swarm_cruise_0.2
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.9 --name swarm_cruise_0.9
python tools/run_experiment.py swarm --seconds 120 --set gain=1.0 --name swarm_gain_1.0
python tools/run_experiment.py swarm --seconds 120 --set gain=10.0 --name swarm_gain_10.0
python tools/run_experiment.py swarm --seconds 120 --set noise=0.3 --name swarm_noise_0.3

# Pioneer Part B Runs (180 s)
python tools/run_experiment.py pioneer --seconds 180 --name pioneer_default
python tools/run_experiment.py pioneer --seconds 180 --set bubble=0.15 --name pioneer_bubble_0.15
python tools/run_experiment.py pioneer --seconds 180 --set bubble=0.60 --name pioneer_bubble_0.60

# Exercise 1: Speed vs. Safety Trade-Off Sweep
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.2
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.4
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.6
python tools/run_experiment.py swarm --seconds 120 --set cruise=0.8
python tools/run_experiment.py swarm --seconds 120 --set cruise=1.0

# Trajectory Visualization and Metric Analysis
python analysis/plot_trajectories.py analysis/runs/swarm_default_rep1.csv
python analysis/plot_trajectories.py analysis/runs/pioneer_default.csv
```

## Generated Figures in `figs/`
- `fig7_1_swarm_arena.png`: High-resolution 3D rendering screenshot of the Webots simulation running in the swarm arena
- `fig7_2_swarm_trajectories.png`: Trajectories of the 5 e-puck robots and 10 cm grid floor coverage heatmap (120 s)
- `fig7_3_pioneer_trajectory.png`: Pioneer 3-DX lidar-guided path and 50 cm grid room coverage heatmap (180 s)
- `fig7_4_exercise1_speed_safety.png`: Plot of mean arena coverage and worst-robot stuck percentage against cruise speed parameter

