# HW3 — Function Approximation RL for Cart-Pole

## Status: Code fixes done, paper drafted. NEEDS RETRAINING before submission.

---

## Context

This is HW3 for FRA503 Deep Reinforcement Learning. We implement 8 RL algorithms
(Linear Q, DQN, MC REINFORCE, AC, A2C, PPO, SAC, TD3) for cart-pole stabilization
using 256 parallel Isaac Lab environments, 20,000 batch steps each.

Two external reviewers audited the paper and code. Their findings drove all changes below.

---

## What Was Found (Problems in Original Submission)

### 1. Deployment Data in Paper Was Wrong

The old `main.tex` Section 3 had fabricated/incorrect deployment numbers.
The actual CSV data (`experiments/suite_1_baseline/*_deploy.csv`) shows:

| Algorithm | Paper Claimed | Actual CSV (100 eps) |
|-----------|--------------|---------------------|
| TD3       | 298.22       | **997.15** (all perfect!) |
| DQN       | 358.10       | 715.48 |
| REINFORCE | 540.67       | 388.59 |
| Linear Q  | 11.39        | 22.44 |
| AC/PPO/A2C/SAC | ~999   | ~999 (roughly correct) |

The paper also said "10 episodes" but the CSVs have **100 episodes**.

**The entire TD3 "brittleness" narrative was false.** TD3 deploys at 997, not 298.
The old Section 4.3 ("Extrinsic vs Intrinsic Exploration") was built on this error.

### 2. Three Algorithm Implementations Were Non-Textbook

Verified by cross-referencing code against Sutton & Barto, lecture PDFs, and original papers:

| Algorithm | Issue | Textbook Says | Code Did |
|-----------|-------|---------------|----------|
| **DQN** | Target network update | Hard copy every C steps (Mnih 2015) | Polyak soft update (DDPG-style) |
| **AC** | Advantage computation | TD error delta_t (S&B Ch 13.5) | MC returns G_t (= REINFORCE with baseline) |
| **MC REINFORCE** | Parallel path baseline | Use V(s) baseline | Dropped baseline entirely (bug) |

### 3. Other Paper Issues
- Performance hierarchy was wrong (TD3 ranked below DQN)
- MC REINFORCE "catastrophic collapse from 997 to 48" was exaggerated (rolling-50 never exceeded 500)
- UTD ratio stated as 4 but code does 64 gradient steps / 256 envs = 0.25 effective

---

## What Was Fixed

### Code Changes (REQUIRE RETRAINING)

#### 1. DQN: Hard Target Update (`RL_Algorithm/Function_based/DQN.py`)
- **Before:** `target_param.data.copy_(tau * param + (1-tau) * target_param)` (Polyak averaging)
- **After:** `if update_count % target_update_freq == 0: target_net.load_state_dict(policy_net.state_dict())`
- Parameter `tau` replaced with `target_update_freq` (default 1000)
- Config updated: `rl_config.json` DQN section: `"tau": 0.005` -> `"target_update_freq": 1000`
- `run_experiment.py` updated to pass `target_update_freq` instead of `tau`
- `scripts/Function_based/train.py` template comment updated

#### 2. AC: TD Error Instead of MC Returns (`RL_Algorithm/Function_based/AC.py`)
- **Before:** `compute_returns()` computed full MC return G_t, advantage = G_t - V(s)
- **After:** `compute_td_errors()` computes delta_t = r + gamma*V(s')*(1-done) - V(s)
- `generate_trajectory()` now also collects `next_values` and `dones` for bootstrapping
- `calculate_loss()` uses TD error as advantage, critic loss = delta_t^2
- Both single-env and parallel-env paths updated
- This makes it a TRUE actor-critic (S&B Ch 13.5) instead of REINFORCE-with-baseline

#### 3. MC REINFORCE: Parallel Baseline Fix (`RL_Algorithm/Function_based/MC_REINFORCE.py`)
- **Before:** `calculate_loss(returns, lp)` — no states passed, baseline skipped
- **After:** `calculate_loss(returns, lp, states)` — states collected per-env, baseline active
- Added `env_states` list to store observations per environment during rollout

#### 4. SAC: Alpha Logging (`RL_Algorithm/Function_based/SAC.py`)
- `update_policy()` return dict now includes `"alpha": self.alpha`
- `run_experiment.py` loss CSV now has `alpha` column
- Needed for Fig 7 (SAC temperature dynamics plot)

### Paper Changes (`main.tex`)

#### Section 2 (Experimental Setup) — minor updates:
- Table 2: DQN row now shows `hard/1000` instead of `tau=0.005`
- Table 2: SAC/TD3 show `64` gradient steps per env step (was `4`)
- Text: "100 episodes" instead of "10 episodes"

#### Sections 3-4 — COMPLETELY REWRITTEN with new structure:
- **Section 3.1:** Training Dynamics — Fig 1 (learning curves) + Fig 2 (convergence bar)
- **Section 3.2:** Deployment Robustness — Table 3 (corrected) + Fig 3 (boxplot)
- **Section 3.3:** Learned Representations — Fig 4 (policy/value surfaces for PPO/SAC/DQN)
- **Section 4.1:** Capacity Boundary — Linear Q vs DQN + Fig 8 (bar chart)
- **Section 4.2:** Precision Boundary — DQN vs continuous + Fig 5 (action traces)
- **Section 4.3:** Variance Boundary — REINFORCE vs AC/A2C/PPO + Fig 6 (variance analysis)
- **Section 4.4:** Exploration Paradigms — TD3 vs SAC + Fig 7 (SAC temperature)
- **Section 4.5:** Implementation Insights — log-std, DQN fix, AC fix, REINFORCE fix, UTD ratio

#### Corrected hierarchy:
```
SAC ~ PPO ~ AC ~ A2C ~ TD3 >> DQN >> MC REINFORCE >> Linear Q
```

#### Placeholder values:
All data-dependent numbers use `\placeholder{...}` (renders as red text in PDF).
These MUST be filled in after retraining. Search for `\placeholder` to find them all.

### New Files Created

#### Plotting Script (`scripts/Function_based/visualize/plot_paper_figures.py`)
Generates all 8 publication figures from CSV data and saved models:
- `fig1_learning_curves.png` — rolling-50/200 return vs batch step, convergence markers
- `fig2_convergence_speed.png` — horizontal bar chart, batch steps to reach 900
- `fig3_deployment_boxplot.png` — box+scatter, color-coded tiers (green/yellow/red)
- `fig4_policy_value_surfaces.png` — 2D heatmaps for PPO/SAC/DQN
- `fig5_action_traces.png` — DQN staircase vs PPO smooth, with theta overlay and RMS
- `fig6_reinforce_variance.png` — REINFORCE scatter+band vs PPO reference
- `fig7_sac_temperature.png` — dual y-axis: alpha + entropy over training
- `fig8_capacity_boundary.png` — Linear / Deep+Discrete / Deep+Continuous bars

---

## What Needs To Be Done Next

### Step 1: RETRAIN all 8 algorithms

```bash
cd CartPole_4.5.0
python run_experiment.py --task Stabilize-Isaac-Cartpole-v0 --headless
```

This will:
- Train all 8 algorithms sequentially (256 parallel envs, 20k batch steps each)
- Deploy each for 100 deterministic episodes
- Save CSVs to `experiments/suite_1_baseline/`
- Save models to `model/Stabilize/{algo}/`
- Log losses (including SAC alpha) to `*_losses.csv`
- Log deployment trajectories to `*_deploy_trajectory.csv`

**Important:** The code fixes change algorithm behavior. Old results in the CSVs are
from the pre-fix implementations and MUST NOT be used in the paper.

### Step 2: Generate figures

```bash
python scripts/Function_based/visualize/plot_paper_figures.py
```

Reads from `experiments/suite_1_baseline/` and `model/Stabilize/`, outputs to `figures/`.

### Step 3: Fill in placeholder values in main.tex

After retraining, compute these from the new CSVs:

1. **Table 3 (deployment):** Mean return, std, episode length range for each algorithm
   - Source: `{algo}_deploy.csv`

2. **Section 3.1 text:**
   - `PPO_conv_step` — batch step where PPO rolling-50 first >= 900
   - `REINFORCE_std_ratio` — ratio of rolling-50 std (REINFORCE / PPO)
   - Source: `{algo}.csv`, compute rolling-50 mean/std over episodes

3. **Section 4.2 text:**
   - `DQN_rms_theta` — RMS of pole angle in last 500 steps of deployment
   - `PPO_rms_theta` — same for PPO
   - Source: `{algo}_deploy_trajectory.csv`, use pole_angle column

4. **Section 4.3 text:**
   - `variance_ratio` — same as REINFORCE_std_ratio above

Search for `\placeholder` in main.tex to find all spots.

### Step 4: Compile and review

```bash
pdflatex main.tex
```

Check that:
- All `\placeholder` markers are gone (no red text)
- Figure files match the `\includegraphics` names
- Table numbers match the narrative
- The hierarchy in Section 4 intro matches the actual deployment data

---

## Key Decisions and Rationale

### Why not keep Polyak update for DQN?
The lecture and Mnih 2015 define DQN with hard target copies. Reviewers flagged this as
a deviation. Since the assignment says "implement DQN," it should match the canonical
definition. Polyak averaging is correct for TD3/SAC but is a DDPG-era convention.

### Why change AC from MC returns to TD error?
Sutton & Barto (Ch 13.4 vs 13.5) explicitly distinguishes:
- Ch 13.4: REINFORCE with baseline = uses G_t - V(s) = MC returns
- Ch 13.5: Actor-Critic = uses delta_t = r + gamma*V(s') - V(s) = TD error

The old AC was Ch 13.4, not Ch 13.5. The whole point of "actor-critic" is that
bootstrapping via V(s') reduces variance compared to full MC returns.

### Why does A2C use lambda=1.0?
A2C's GAE with lambda=1.0 makes it functionally similar to MC returns
(the telescoping sum recovers G_t - V(s_t)). This is the SB3 default.
The key difference from AC: A2C uses fixed-length rollouts (T=4) and
bootstraps at the rollout boundary via V(s_T), while AC waits for episode end.
This is a valid design choice, not a bug.

### Why is the effective UTD ratio 0.25, not 4?
The code does 64 gradient steps per environment step, but 256 environments each produce
1 transition per step = 256 new transitions. UTD = 64/256 = 0.25.
This is intentional (Raffin 2024 SB3 scaling convention) to prevent overfitting to
the most recent parallel batch.

### Why wasn't TD3 "brittle"?
The old paper's claim that TD3 deploys at 298 was simply wrong data. The actual CSV
shows 997.15. Both TD3 and SAC succeed because cart-pole is simple enough that either
exploration paradigm (fixed noise vs entropy regularization) works. The interesting
analysis is now about mechanisms, not outcomes.

---

## File Map (what was modified)

```
CartPole_4.5.0/
  RL_Algorithm/Function_based/
    DQN.py                    # Hard target update (was Polyak)
    AC.py                     # TD error advantage (was MC returns)
    MC_REINFORCE.py           # Parallel baseline fix
    SAC.py                    # Added alpha to loss dict
  run_experiment.py           # DQN uses target_update_freq; SAC logs alpha
  scripts/Function_based/
    configs/rl_config.json    # DQN: tau -> target_update_freq
    train.py                  # Template comment updated
    visualize/
      plot_paper_figures.py   # NEW: generates all 8 paper figures
      plot_report_figures.py  # OLD: original plotting script (still works)
  main.tex                    # Sections 2-4 rewritten, placeholders added
  figures/                    # Will be populated by plot_paper_figures.py
  experiments/suite_1_baseline/  # Will be populated by retraining
```

---

## Reviewer Audit Summary

### Sections 1-2: Verified correct (with minor fixes applied)
- All 8 equations match their source papers / S&B / lectures
- Taxonomy table correct given the implementations
- Only fixes: DQN target update method, deployment episode count, UTD ratio

### Algorithms verified clean (no code changes needed):
- Linear Q: matches S&B Ch 10, lecture 7
- A2C: GAE implementation correct, lambda=1.0 is a valid choice
- PPO: matches Schulman 2017 exactly
- SAC: matches Haarnoja 2018b exactly (twin critics, reparameterization, auto-alpha)
- TD3: matches Fujimoto 2018 exactly (twin critics, delayed actor, target smoothing)

### Algorithms with non-standard additions (documented, not bugs):
- MC REINFORCE: has entropy bonus (not in Williams 1992 or S&B, but valid extension)
- AC: has entropy bonus (same note)
