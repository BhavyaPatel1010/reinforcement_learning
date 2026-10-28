# Grocery Shopping Solution: Supply-Chain Inventory Management with Reinforcement Learning

This project learns a daily grocery replenishment policy from historical demand. The agent decides **how many units to order**, while delayed deliveries, holding inventory, and stock-outs create real inventory trade-offs.

## Methods implemented

| Method | Purpose |
|---|---|
| Reorder-point policy | Non-RL business baseline for a fair comparison |
| Tabular Q-learning | Explainable RL method using discretized states |
| Deep Q-Network (DQN) | Main RL method using a neural network and replay buffer |

The dataset is used to replay actual demand; it is **not** used as a prediction target. This is a decision/control problem, which makes it appropriate for reinforcement learning.

## Environment design

- **State:** inventory on hand, outstanding/pipeline orders, 3 recent demand values, 7-day demand mean, normalized price, promotion, and weekday.
- **Actions:** order 0, 5, 10, ..., 50 units per day.
- **Lead time:** default two days (configurable).
- **Reward:** sales margin - ordering cost - holding cost - lost-sales penalty.
- **Episode:** a chronological demand sequence. The final 15% of observations is never used for training and is reserved for testing.

## Kaggle instructions

1. Create a new Kaggle Notebook and enable **GPU T4 x2** only if you want faster DQN training. CPU is sufficient for the default settings.
2. Add a CSV dataset to the notebook. The recommended choice is Kaggle's *Retail Store Inventory and Demand Forecasting* dataset.
3. Import this GitHub repository in Kaggle, or upload/download it as a dataset. In a notebook cell:

```python
!git clone https://github.com/BhavyaPatel1010/reinforcement_learning.git
%cd reinforcement_learning
!pip install -q -r requirements.txt
```

4. Locate your input CSV:

```python
!find /kaggle/input -name "*.csv" | head -20
```

5. Train both RL models (replace the path with the path shown above):

```python
!python train.py \
  --data-path /kaggle/input/retail-store-inventory-and-demand-forecasting/retail_store_inventory.csv \
  --q-episodes 450 \
  --dqn-episodes 220 \
  --output-dir /kaggle/working/results
```

If the data has more than one product and store, select one pair first:

```python
!python train.py --data-path /kaggle/input/YOUR-DATASET/YOUR-FILE.csv --product-id 101 --store-id 1 --output-dir /kaggle/working/results
```

6. Inspect outputs:

```python
import pandas as pd
from IPython.display import display, Image
display(pd.read_csv('/kaggle/working/results/comparison_metrics.csv'))
display(Image('/kaggle/working/results/training_rewards.png'))
```

## Local smoke test

No data file is required for a basic run; a realistic seasonal synthetic-demand series will be generated:

```bash
pip install -r requirements.txt
python train.py --q-episodes 100 --dqn-episodes 50
```

## CSV compatibility

The loader recognizes common column names:

- demand: `Demand`, `Units Sold`, `sales`, or `Demanda_uni_equil`
- product: `Product ID`, `item`, `item_id`, or `Producto_ID`
- store: `Store ID`, `store`, `store_nbr`, or `Agencia_ID`
- optional price: `Price` or `sell_price`; optional promotion: `Promotion`, `onpromotion`, or `Discount`

If your file uses different names, change the lists in `src/data.py`.

## Submission/report checklist

1. Explain the business issue: stock-outs lose sales; overstock raises holding/waste cost.
2. Explain the state, action, transition, and reward above.
3. Compare reorder-point, Q-learning, and DQN on the held-out test period.
4. Include `comparison_metrics.csv`, `training_rewards.png`, and one daily log plot/table.
5. State limitations: simulated lead time/costs and a single selected product-store pair.

## Project structure

```text
train.py                Kaggle entry point: trains and evaluates all methods
src/inventory_env.py    Inventory simulator and reward function
src/agents.py           Q-learning and DQN implementations
src/data.py             Kaggle CSV loader and synthetic smoke-test data
outputs/                Created at runtime; metrics, logs, plot, metadata
```
