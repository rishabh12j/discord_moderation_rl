# Early prototype (February 2026)

The first version of the Discord moderation RL project, kept for its history and tests. The final version, published at MURL 2026, is at the top level of this repository.

## What is here
| Folder | Contents |
|---|---|
| `src/env/` | First Gymnasium environment (`DiscordEnv`), environment factory and the Lagrangian reward wrapper |
| `src/utils/` | Episode builder, toxicity judge, user features, data download and setup checks |
| `scripts/` | Baseline evaluation, reward analysis and sensitivity, model comparison, synthetic data generation, demos |
| `tests/` | 12 test files: action masking, environment, state transitions, wrappers, episode builder, toxicity judge, user features and SB3 compatibility |
| `configs/` | Environment and toxicity judge settings |
| `data/processed/` | Comments from the public Civil Comments dataset, grouped into conversations with simulated user profiles |

## Running it
Paths are relative to this folder, so run everything from here:
```bash
cd archive/early-prototype
pip install -r requirements.txt
python -m pytest tests
```
This code is not maintained. Some modules referenced by the scripts were reworked or renamed in the final version.
