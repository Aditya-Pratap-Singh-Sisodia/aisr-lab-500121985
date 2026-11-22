# Experiment 8: Cross-Embodiment Transfer of a Manipulation Skill (PyBullet)

## Environment and Platform
- Python: 3.12 / 3.11 with PyBullet, NumPy, Matplotlib
- Operating System: Windows 11 (MSI / x86_64)
- Embodiments Tested:
  1. KUKA LBR iiwa (7-DOF, 0.94 m reach)
  2. Franka Emika Panda (7-DOF, 0.94 m reach)
  3. UFactory xArm6 (6-DOF, 0.76 m reach)
  4. KUKA iiwa scaled to 70% (kuka_mini, 7-DOF, 0.66 m reach)

## Commands Executed
```bash
# Verify setup and load robot URDF models
python experiments/00_check_setup.py

# Measure kinematics, reach, and workspace volume
python experiments/01_explore_embodiments.py --samples 3000

# Joint-space naive replay of KUKA trajectories
python experiments/02_joint_space_transfer.py --episodes 10

# Task-space expert replay with per-robot Damped Least-Squares IK
python experiments/03_task_space_transfer.py --episodes 20

# Learn once on KUKA (BC-agnostic vs BC-proprio) and deploy on all robots
python experiments/04_learn_once_deploy_many.py --demos 60 --episodes 20

# Multi-embodiment training and leave-one-out evaluation
python experiments/05_multi_embodiment.py --demos 60 --episodes 20
```

## Generated Figures in `figs/`
- `01_workspace.png`: Workspace reach and task-space coverage of the four arms
- `02_joint_space_paths.png`: End-effector paths under naive joint-space replay
- `03_task_space_paths.png`: End-effector paths under task-space transfer
- `04_learn_once_deploy_many_kuka_iiwa.png`: Success rates of BC-agnostic vs BC-proprio across embodiments
- `05_multi_embodiment.png`: Leave-one-out cross-embodiment evaluation results

## Summary of Results
1. **Joint Space vs Task Space**: Naive joint angle transfer failed completely on all target robots (0% success) due to kinematic differences. Task-space transfer succeeded 100% on KUKA, Panda, and xArm6.
2. **Observation Gap**: BC-proprio (trained with joint angles) failed 0% when transferred across embodiments. BC-agnostic (using only end-effector pose deltas) transferred zero-shot with 100% success on full-reach arms.
3. **Capability Gap**: On `kuka_mini`, success was bounded at ~50% across both expert and learned policies because 52% of cube targets lay beyond the robot's physical reach envelope (0.66 m vs 0.94 m).

