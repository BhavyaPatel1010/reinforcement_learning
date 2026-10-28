"""Train and compare Q-learning, DQN, and a reorder-point baseline.

Example Kaggle command:
!python train.py --data-path /kaggle/input/your-dataset/file.csv --dqn-episodes 220
"""
from __future__ import annotations
import argparse, json, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import load_retail_data
from src.inventory_env import InventoryEnv, InventoryConfig
from src.agents import QLearningAgent, DQNAgent


def run_policy(env, policy, explore=False):
    state = env.reset(); done = False; total = 0.; rows = []
    while not done:
        action = policy(state, explore) if callable(policy) else policy.act(state, explore)
        state, reward, done, info = env.step(action); total += reward; rows.append({**info, "reward": reward, "action": action})
    return total, pd.DataFrame(rows)


def train_q(env, episodes):
    agent = QLearningAgent(env.action_size)
    rewards = []
    for _ in range(episodes):
        total, _ = run_policy(env, agent, True)
        # replay one more episode while updating action values
        state = env.reset(); done = False
        while not done:
            a = agent.act(state, True); nxt, r, done, _ = env.step(a); agent.learn(state, a, r, nxt, done); state = nxt
        agent.decay(); rewards.append(total)
    return agent, rewards


def train_dqn(env, episodes, device):
    agent = DQNAgent(env.state_size, env.action_size, device=device)
    rewards = []
    for _ in range(episodes):
        state = env.reset(); done = False; total = 0.
        while not done:
            a = agent.act(state, True); nxt, r, done, _ = env.step(a)
            agent.remember(state, a, r, nxt, done); agent.learn(); state = nxt; total += r
        agent.decay(); rewards.append(total)
    return agent, rewards


def reorder_point_policy(env):
    threshold = env.mean_demand * (env.config.lead_time + 1.5)
    def choose(state, _):
        inventory_position = state[0] * env.max_inventory + state[1] * env.max_inventory
        return env.action_size - 1 if inventory_position < threshold else 0
    return choose


def metrics(name, reward, log):
    demand, sales = log.demand.sum(), log.sales.sum()
    return {"method": name, "total_reward": round(reward, 2), "fill_rate": round(float(sales / max(demand, 1)), 4),
            "stockout_rate": round(float(log.lost_sales.sum() / max(demand, 1)), 4),
            "average_inventory": round(float(log.inventory.mean()), 2), "total_order_cost": round(float(log.order_cost.sum()), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path"); p.add_argument("--product-id"); p.add_argument("--store-id")
    p.add_argument("--output-dir", default="outputs"); p.add_argument("--q-episodes", type=int, default=450); p.add_argument("--dqn-episodes", type=int, default=220)
    p.add_argument("--lead-time", type=int, default=2); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    data, source = load_retail_data(args.data_path, args.product_id, args.store_id)
    n = len(data); train_end, test_start = int(n * .70), int(n * .85)
    train, test = data.iloc[:train_end], data.iloc[test_start:]
    cfg = InventoryConfig(lead_time=args.lead_time)
    train_env = InventoryEnv(train.demand, train.price, train.promotion, cfg)
    test_env = InventoryEnv(test.demand, test.price, test.promotion, cfg)
    q_agent, q_rewards = train_q(train_env, args.q_episodes)
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
        dqn_agent, dqn_rewards = train_dqn(train_env, args.dqn_episodes, device)
    except ImportError as exc: raise RuntimeError("Install PyTorch before DQN training: pip install torch") from exc
    experiments = [("Reorder point", reorder_point_policy(test_env)), ("Q-Learning", q_agent), ("DQN", dqn_agent)]
    rows = []
    for name, agent in experiments:
        reward, log = run_policy(test_env, agent, False); rows.append(metrics(name, reward, log)); log.to_csv(f"{args.output_dir}/{name.lower().replace(' ', '_')}_daily_log.csv", index=False)
    report = pd.DataFrame(rows); report.to_csv(f"{args.output_dir}/comparison_metrics.csv", index=False)
    with open(f"{args.output_dir}/run_metadata.json", "w") as f: json.dump({"data_source": source, "train_rows": len(train), "test_rows": len(test), "lead_time": args.lead_time}, f, indent=2)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(q_rewards, label="Q-learning", alpha=.75); ax.plot(np.linspace(0, len(q_rewards)-1, len(dqn_rewards)), dqn_rewards, label="DQN", alpha=.75); ax.set(title="Training reward", xlabel="Episode", ylabel="Reward"); ax.legend(); fig.tight_layout(); fig.savefig(f"{args.output_dir}/training_rewards.png", dpi=150)
    print("Data:", source); print("\\nTEST-SET COMPARISON\\n", report.to_string(index=False)); print(f"\\nSaved results to {args.output_dir}/")

if __name__ == "__main__": main()
