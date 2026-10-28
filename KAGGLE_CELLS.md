# Kaggle notebook: copy and run these cells in order

You do **not** need to search for or manually attach a dataset. The training
command downloads the exact public Kaggle dataset using this handle:

`atomicd/retail-store-inventory-and-demand-forecasting`

## Cell 1 — clone the project

```python
!git clone https://github.com/BhavyaPatel1010/reinforcement_learning.git
%cd /kaggle/working/reinforcement_learning
```

If the folder already exists after rerunning the notebook, use this instead:

```python
%cd /kaggle/working/reinforcement_learning
!git pull
```

## Cell 2 — install requirements

```python
!pip install -q -r requirements.txt
```

## Cell 3 — quick end-to-end test

```python
!python train.py \
    --kaggle-dataset \
    --q-episodes 10 \
    --dqn-episodes 5 \
    --output-dir /kaggle/working/test_results
```

## Cell 4 — full training

The action quantities automatically scale to the selected product's average
demand, so no manual inventory tuning is required.

```python
!python train.py \
    --kaggle-dataset \
    --q-episodes 450 \
    --dqn-episodes 220 \
    --lead-time 2 \
    --output-dir /kaggle/working/results
```

## Cell 5 — show the comparison and training curve

```python
import pandas as pd
from IPython.display import Image, display

results = pd.read_csv("/kaggle/working/results/comparison_metrics.csv")
display(results)
display(Image("/kaggle/working/results/training_rewards.png"))
```

## Cell 6 — plot the DQN decisions

```python
import matplotlib.pyplot as plt

dqn = pd.read_csv("/kaggle/working/results/dqn_daily_log.csv")
plt.figure(figsize=(15, 6))
plt.plot(dqn["demand"], label="Demand")
plt.plot(dqn["inventory"], label="Remaining inventory")
plt.plot(dqn["order_qty"], label="DQN order quantity", alpha=0.75)
plt.title("DQN grocery inventory decisions on unseen test data")
plt.xlabel("Test day")
plt.ylabel("Units")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

## Cell 7 — package results for submission

```python
!cd /kaggle/working && zip -r grocery_rl_results.zip results
```

Finally select **Save Version → Save & Run All** in Kaggle.
