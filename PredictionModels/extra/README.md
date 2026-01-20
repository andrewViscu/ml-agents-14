# Extra Analysis Scripts

Additional scripts exploring alternative approaches and extended analysis beyond the main RQ1/RQ2.

## Scripts

### cutoff_analysis.py

Tests the original RQ2 approach: predicting final reward from early training data at different step cutoffs.

```bash
python cutoff_analysis.py --training_data ../../training-data --shared_data ../../shared-data
```

**Results (423 runs with time-series data):**

| Cutoff | Runs | R² |
|--------|------|-----|
| 100,000 | 423 | 0.10 |
| 200,000 | 423 | 0.21 |
| 300,000 | 338 | 0.25 |
| 400,000 | 337 | 0.37 |
| 500,000 | 227 | 0.45 |
| 600,000 | 225 | 0.51 |
| 800,000 | 225 | 0.56 |
| 1,000,000 | 131 | 0.73 |

More early training data improves predictions, but requires waiting longer before prediction is useful.

### all_games_reward_prediction.py

Extends RQ2 to all available games to test if single-agent vs multi-agent environments differ in predictability.

```bash
python all_games_reward_prediction.py --shared_data ../../shared-data
```

**Results:**

| Game | Type | Runs | R² | Reward Range |
|------|------|------|-----|--------------|
| 3DBall | Single | 4390 | 0.77 | 99.2 |
| GridFoodCollector | Multi | 198 | 0.56 | 72.0 |
| PushBlock | Multi | 192 | 0.53 | 6.0 |
| Crawler | Single | 356 | 0.42 | 2195.4 |
| Basic | Single | 46 | 0.14 | 0.8 |
| SoccerTwos | Multi | 142 | 0.08 | 0.1 |
| Hallway | Multi | 49 | -0.33 | 1.4 |

**Excluded:** BigWallJump (no HP variation), GridWorld/Pyramids/Sorter/Walker/Worm (< 30 runs)

**Findings:**

Predictability depends on:
1. **Sample size** - need 30+ runs for reliable results
2. **Reward range** - narrow ranges (SoccerTwos: 0.1, Hallway: 1.4) yield poor R²
3. **HP variation** - identical hyperparameters make prediction impossible

Single vs multi-agent distinction is not the determining factor.
