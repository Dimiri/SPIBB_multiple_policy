# Literature Matrix

| # | Paper | Year | Main Topic | Baseline | Factored | Robust | Bootstrapping | Main Relevance |
|---|---|---:|---|:---:|:---:|:---:|:---:|---|
| 1 | Laroche et al. — Safe Policy Improvement with Baseline Bootstrapping | 2019 | SPIBB | ✓ | ✗ | ✗ | ✓ | Original SPIBB framework |
| 2 | Simão & Spaan — Safe Policy Improvement with Baseline Bootstrapping in Factored Environments | 2019 | Factored SPIBB | ✓ | ✓ | ✗ | ✓ | Main method we want to extend |
| 3 | Iyengar — Robust Dynamic Programming | 2005 | Robust MDPs | ✗ | ✗ | ✓ | ✗ | Foundations of robust dynamic programming |
| 4 | Nilim & El Ghaoui — Robust Control of Markov Decision Processes with Uncertain Transition Matrices | 2005 | Robust MDPs | ✗ | ✗ | ✓ | ✗ | Robust transition uncertainty and ambiguity sets |
| 5 | Petrik et al. — Safe Policy Improvement by Minimizing Robust Baseline Regret | 2016 | Robust safe policy improvement | ✓ | ✗ | ✓ | ✗ | Connects robust uncertainty with baseline-relative improvement |
| 6 | Schnitzer et al. — Efficient Solution and Learning of Robust Factored MDPs | 2026 | Robust Factored MDPs | ✗ | ✓ | ✓ | ✗ | Connects robust ambiguity with factored structure |
| 7 | Strehl — Model-Based Reinforcement Learning in Factored-State MDPs | 2007 | Factored MDPs | ✗ | ✓ | ✗ | ✗ | Factored transition estimation and error composition |
| 8 | Galesloot et al. — Robust Probabilistic Shielding for Safe Offline Reinforcement Learning | 2026 | Robust safe offline RL | ✓ | ✗ | ✓ | ✗ | Recent adjacent work; important for differentiation |
| 9 | Weissman et al. — Inequalities for the L1 Deviation of the Empirical Distribution | 2003 | Concentration bounds | ✗ | ✗ | ✗ | ✗ | Mathematical basis for L1 estimation-error bounds |
| — | **Proposed: Robust Factored SPIBB** | — | Robust Factored SPI | ✓ | ✓ | ✓ | ✓ | **Proposed research gap** |

## Legend

- **Baseline** — explicitly evaluates/improves relative to an existing policy
- **Factored** — exploits factored state/transition structure
- **Robust** — explicitly represents transition-model ambiguity
- **Bootstrapping** — constrains uncertain decisions toward the baseline policy
