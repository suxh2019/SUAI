# Laptop Robot RL Research Prototype

An **educational research scaffold**, not a production robot-control system.
It demonstrates several complementary ideas with a CPU-only 2D continuous-control simulator.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick verification

```bash
python -m unittest -v test_robot_rl.py
python robot_rl_research.py --quick
```

## More complete experiment

```bash
python robot_rl_research.py --seeds 11 22 --sizes 60 120 --widths 32 64
```

The default run performs 2 seeds × 2 dataset sizes × 2 model widths, and may take significantly longer than `--quick` on a CPU. For a modest intermediate run:

```bash
python robot_rl_research.py --seeds 11 --sizes 60 --widths 32 --inverse-steps 150 --world-steps 250 --critic-steps 200 --bc-steps 200
```

## Files and outputs

- `robot_rl_research.py`: fully commented environment, data, models, fitting, and evaluation.
- `test_robot_rl.py`: basic correctness/regression checks.
- `requirements.txt`: only PyTorch and NumPy.
- `robot_rl_outputs/results.csv`: machine-readable per-method metrics.
- `robot_rl_outputs/config.json`: exact experiment settings.
- `robot_rl_outputs/actor_*.pt`: actor checkpoints.

## Research hypotheses and limitations

1. **Mixed-quality demonstrations**: compare behavior cloning with clean expert actions versus all available actions. Data quality can outweigh sheer volume.
2. **Videos without labels**: infer *pseudo-actions* from consecutive states using a supervised inverse-dynamics model and try those pseudo-actions in training. These are **state trajectories**, not real RGB videos or a computer vision pipeline. Quality of inferred actions can be poor.
3. **Sparse rewards**: learn a Q-function using a sparse terminal success signal, TD bootstrapping, and a **fixed expert next-action policy**. This is policy evaluation, **not** a fully learned sparse-reward RL agent and does not guarantee goal-reaching from offline data.
4. **World model + MPC**: regress next-state changes then use random-shooting receding-horizon planning. This model can fail from compounding errors and weak action coverage. A horizon of 7 may be too short for a distant goal; the default planner is intentionally simple.
5. **Scale and reproducibility**: vary data episodes and hidden widths, repeat with independent seeds, preserve settings, evaluate on separate starts, and save outputs.

## Interpreting results

- Success = reaching distance less than 0.16 from the goal within 35 steps.
- `mean_length` is average episode length.
- `world_model_mse` is an **in-sample diagnostic**, NOT a held-out dynamics test. Add separate validation trajectories for publication-quality results.
- `mean_offline_q` is a diagnostic value estimate, NOT proof of accurate return prediction.
- Small sample sizes can produce noisy estimated success rates.
- Never control real physical hardware with this demo without extensive safety engineering.

## Research extensions

- Replace state-vector observation sequences with actual video frames and a visual encoder.
- Add inverse-dynamics validation and confidence filtering of inferred actions.
- Train Q-functions using n-step targets, conservative offline RL, and held-out calibration.
- Learn a model ensemble and add uncertainty-aware MPC / CEM planning.
- Add fair wall-clock and interaction-budget comparisons with several seeds and confidence intervals.
