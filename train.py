"""Train and compare Q-learning, DQN, and a reorder-point baseline.

Example Kaggle command:
!python train.py --data-path /kaggle/input/your-dataset/file.csv --dqn-episodes 220
"""
from __future__ import annotations
import argparse, json, os, random, copy, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data import DEFAULT_KAGGLE_DATASET, download_kaggle_dataset, load_retail_data
from src.inventory_env import InventoryEnv, InventoryConfig
from src.agents import QLearningAgent, DQNAgent


def run_policy(env, policy, explore=False):
    state = env.reset(); done = False; total = 0.; rows = []
    while not done:
        action = policy(state, explore) if callable(policy) else policy.act(state, explore)
        state, reward, done, info = env.step(action); total += reward; rows.append({**info, "reward": reward, "action": action})
    return total, pd.DataFrame(rows)


def train_q(env, episodes, validation):
    agent = QLearningAgent(env.action_size)
    rewards = []
    best_score, best = -float('inf'), None
    for episode in range(episodes):
        state = env.reset(); done = False; total = 0.0
        while not done:
            a = agent.act(state, True); nxt, r, done, _ = env.step(a)
            agent.learn(state, a, r, nxt, done); state = nxt; total += r
        agent.decay(); rewards.append(total)
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            score, _ = run_policy(validation, agent)
            if score > best_score:
                best_score, best = score, copy.deepcopy(agent.q)
            print(f'Q-learning {episode+1}/{episodes}: validation reward={score:.2f}', flush=True)
    agent.q = best
    return agent, rewards


def train_dqn(env, episodes, device, validation):
    agent = DQNAgent(env.state_size, env.action_size, device=device)
    rewards = []
    agent.epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / max(1, episodes * .8))
    best_score, best = -float('inf'), None
    for episode in range(episodes):
        state = env.reset(); done = False; total = 0.
        while not done:
            a = agent.act(state, True); nxt, r, done, _ = env.step(a)
            agent.remember(state, a, r, nxt, done); agent.learn(); state = nxt; total += r
        agent.decay(); rewards.append(total)
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            score, _ = run_policy(validation, agent)
            if score > best_score:
                best_score, best = score, copy.deepcopy(agent.net.state_dict())
            print(f'Double DQN {episode+1}/{episodes}: validation reward={score:.2f}', flush=True)
    agent.net.load_state_dict(best)
    return agent, rewards


def reorder_point_policy(env):
    threshold = env.mean_demand * (env.config.lead_time + 1.5)
    def choose(state, _):
        inventory_position = state[0] * env.max_inventory + state[1] * env.max_inventory
        return env.action_size - 1 if inventory_position < threshold else 0
    return choose


def metrics(name, reward, log):
    demand, sales = log.demand.sum(), log.sales.sum()
    return {"method": name, "total_reward": round(reward, 2), "total_profit": round(float(log.profit.sum()), 2),
            "fill_rate": round(float(sales / max(demand, 1)), 4),
            "stockout_rate": round(float(log.lost_sales.sum() / max(demand, 1)), 4),
            "average_inventory": round(float(log.inventory.mean()), 2), "total_order_cost": round(float(log.order_cost.sum()), 2)}


def save_report_assets(output_dir, report, logs, q_rewards, dqn_rewards):
    """Create report-ready figures, a written summary, and a downloadable ZIP."""
    output = Path(output_dir).resolve()
    colors = {"Reorder point": "#6b7280", "Q-Learning": "#2563eb", "DQN": "#ea580c"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plot_specs = [
        ("total_profit", "Simulated profit", "Profit objective"),
        ("fill_rate", "Demand fill rate", "Fraction of demand"),
        ("average_inventory", "Average inventory", "Units"),
        ("total_order_cost", "Total ordering cost", "Cost"),
    ]
    for ax, (column, title, ylabel) in zip(axes.flat, plot_specs):
        bars = ax.bar(report.method, report[column], color=[colors[m] for m in report.method])
        ax.set(title=title, ylabel=ylabel)
        ax.grid(axis="y", alpha=.25)
        ax.tick_params(axis="x", rotation=12)
        if column == "fill_rate":
            ax.set_ylim(max(0, report[column].min() - .03), 1.005)
            labels = [f"{value:.2%}" for value in report[column]]
        else:
            labels = [f"{value:,.0f}" for value in report[column]]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    fig.suptitle("Inventory policy comparison on unseen test data", fontsize=15)
    fig.tight_layout(); fig.savefig(output / "policy_comparison.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    dqn = logs["DQN"]
    days = np.arange(1, len(dqn) + 1)
    axes[0].plot(days, dqn.demand, label="Demand", color="#111827", linewidth=1.5)
    axes[0].plot(days, dqn.sales, label="Sales fulfilled", color="#16a34a", linewidth=1.2)
    axes[0].fill_between(days, dqn.sales, dqn.demand, where=dqn.demand > dqn.sales,
                         color="#dc2626", alpha=.35, label="Unmet demand")
    axes[0].set(ylabel="Units", title="Double DQN demand fulfilment"); axes[0].legend(ncol=3); axes[0].grid(alpha=.25)
    axes[1].plot(days, dqn.inventory, label="End-of-day inventory", color="#2563eb")
    axes[1].bar(days, dqn.order_qty, label="Order quantity", color="#f97316", alpha=.45)
    axes[1].set(xlabel="Test day", ylabel="Units", title="Double DQN inventory and replenishment")
    axes[1].legend(ncol=2); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "dqn_daily_operations.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, log in logs.items():
        axes[0].plot(np.arange(1, len(log) + 1), log.profit.cumsum(), label=method, color=colors[method])
        axes[1].plot(np.arange(1, len(log) + 1), log.lost_sales.cumsum(), label=method, color=colors[method])
    axes[0].set(title="Cumulative simulated profit", xlabel="Test day", ylabel="Profit objective")
    axes[1].set(title="Cumulative unmet demand", xlabel="Test day", ylabel="Units")
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "cumulative_outcomes.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, log in logs.items():
        counts = log.order_qty.value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", label=method, color=colors[method])
    ax.set(title="Order-quantity distribution", xlabel="Units ordered", ylabel="Number of test days")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / "order_quantity_distribution.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for ax, rewards, name, color in [
        (axes[0], q_rewards, "Q-Learning", colors["Q-Learning"]),
        (axes[1], dqn_rewards, "Double DQN", colors["DQN"]),
    ]:
        series = pd.Series(rewards)
        episodes = np.arange(1, len(series) + 1)
        window = min(20, max(2, len(series) // 10))
        ax.plot(episodes, series, color=color, alpha=.25, label="Episode reward")
        ax.plot(episodes, series.rolling(window, min_periods=1).mean(), color=color,
                linewidth=2, label=f"{window}-episode moving average")
        ax.set(title=f"{name} learning curve", xlabel="Episode", ylabel="Reward")
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output / "learning_curves_smoothed.png", dpi=180); plt.close(fig)

    baseline = report.loc[report.method == "Reorder point"].iloc[0]
    dqn_metrics = report.loc[report.method == "DQN"].iloc[0]
    profit_change = (dqn_metrics.total_profit / baseline.total_profit - 1) * 100
    unmet_reduction = (1 - dqn_metrics.stockout_rate / baseline.stockout_rate) * 100
    inventory_change = (dqn_metrics.average_inventory / baseline.average_inventory - 1) * 100
    markdown_columns = ["Method", "Reward", "Profit", "Fill rate", "Stockout rate", "Avg. inventory", "Order cost"]
    markdown_rows = ["| " + " | ".join(markdown_columns) + " |",
                     "| " + " | ".join(["---"] * len(markdown_columns)) + " |"]
    for row in report.itertuples(index=False):
        markdown_rows.append(
            f"| {row.method} | {row.total_reward:.2f} | {row.total_profit:,.2f} | "
            f"{row.fill_rate:.2%} | {row.stockout_rate:.2%} | {row.average_inventory:.2f} | "
            f"{row.total_order_cost:,.2f} |"
        )
    markdown_table = "\\n".join(markdown_rows)
    summary = f"""# Grocery Inventory RL — Result Summary

## Main finding

On the held-out test period, Double DQN achieved a simulated profit objective of
{dqn_metrics.total_profit:,.2f} and fulfilled {dqn_metrics.fill_rate:.2%} of demand.
Relative to the reorder-point baseline, its profit objective changed by
{profit_change:+.2f}%, unmet demand fell by {unmet_reduction:.2f}%, and average
inventory changed by {inventory_change:+.2f}%.

## Method comparison

{markdown_table}

## Interpretation limits

The profit value is a simulated, penalty-adjusted objective rather than audited
business profit. Results describe one selected product-store series, one temporal
test period, one random seed, and the configured cost assumptions. The dataset is
synthetic. Multi-seed and multi-product evaluation is needed before making a
general business claim.
"""
    (output / "REPORT_SUMMARY.md").write_text(summary, encoding="utf-8")

    figures = ["policy_comparison.png", "dqn_daily_operations.png", "cumulative_outcomes.png",
               "order_quantity_distribution.png", "learning_curves_smoothed.png", "training_rewards.png"]
    manifest = {"tables": ["comparison_metrics.csv"], "figures": figures,
                "models": ["dqn_model.pt", "q_table.json"], "summary": "REPORT_SUMMARY.md"}
    (output / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = shutil.make_archive(str(output), "zip", root_dir=output)
    return archive_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path"); p.add_argument("--product-id"); p.add_argument("--store-id")
    p.add_argument(
        "--kaggle-dataset",
        nargs="?",
        const=DEFAULT_KAGGLE_DATASET,
        help=("Download a Kaggle dataset automatically. Pass this flag without a value "
              f"to use {DEFAULT_KAGGLE_DATASET!r}."),
    )
    p.add_argument("--output-dir", default="outputs"); p.add_argument("--q-episodes", type=int, default=450); p.add_argument("--dqn-episodes", type=int, default=220)
    p.add_argument("--lead-time", type=int, default=2); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    if min(args.q_episodes, args.dqn_episodes) < 1 or args.lead_time < 0:
        p.error('Episodes must be positive and lead time must be nonnegative.')
    os.makedirs(args.output_dir, exist_ok=True)
    data_path = args.data_path
    if args.kaggle_dataset:
        if data_path:
            p.error("Use either --data-path or --kaggle-dataset, not both.")
        data_path = download_kaggle_dataset(args.kaggle_dataset)
    data, source = load_retail_data(data_path, args.product_id, args.store_id)
    n = len(data); train_end, test_start = int(n * .70), int(n * .85)
    train, validation, test = data.iloc[:train_end], data.iloc[train_end:test_start], data.iloc[test_start:]
    cfg = InventoryConfig(lead_time=args.lead_time)
    train_env = InventoryEnv(train.demand, train.price, train.promotion, cfg)
    validation_env = InventoryEnv(validation.demand, validation.price, validation.promotion, cfg, train_env.reference_stats)
    test_env = InventoryEnv(test.demand, test.price, test.promotion, cfg, train_env.reference_stats)
    q_agent, q_rewards = train_q(train_env, args.q_episodes, validation_env)
    try:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(1)
        print(f'Training on {device}', flush=True)
        dqn_agent, dqn_rewards = train_dqn(train_env, args.dqn_episodes, device, validation_env)
    except ImportError as exc: raise RuntimeError("Install PyTorch before DQN training: pip install torch") from exc
    experiments = [("Reorder point", reorder_point_policy(test_env)), ("Q-Learning", q_agent), ("DQN", dqn_agent)]
    rows, logs = [], {}
    for name, agent in experiments:
        reward, log = run_policy(test_env, agent, False); rows.append(metrics(name, reward, log)); logs[name] = log
        log.to_csv(f"{args.output_dir}/{name.lower().replace(' ', '_')}_daily_log.csv", index=False)
    report = pd.DataFrame(rows); report.to_csv(f"{args.output_dir}/comparison_metrics.csv", index=False)
    torch.save({'state_dict': dqn_agent.net.state_dict(), 'state_size': train_env.state_size,
                'action_values': train_env.action_values.tolist(), 'reference_stats': train_env.reference_stats},
               f'{args.output_dir}/dqn_model.pt')
    with open(f'{args.output_dir}/q_table.json', 'w') as f:
        json.dump([{'state': list(k), 'values': v.tolist()} for k, v in q_agent.q.items()], f)
    pd.DataFrame({'episode': range(1, len(q_rewards)+1), 'reward': q_rewards}).to_csv(f'{args.output_dir}/q_training.csv', index=False)
    pd.DataFrame({'episode': range(1, len(dqn_rewards)+1), 'reward': dqn_rewards}).to_csv(f'{args.output_dir}/dqn_training.csv', index=False)
    with open(f"{args.output_dir}/run_metadata.json", "w") as f: json.dump({"data_source": source, "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "lead_time": args.lead_time, "seed": args.seed, "algorithm": "Double DQN", "reference_stats": train_env.reference_stats, "arguments": vars(args)}, f, indent=2)
    fig, ax = plt.subplots(figsize=(9, 4)); ax.plot(range(1, len(q_rewards)+1), q_rewards, label="Q-learning", alpha=.75); ax.plot(range(1, len(dqn_rewards)+1), dqn_rewards, label="Double DQN", alpha=.75); ax.set(title="Training reward", xlabel="Actual episode", ylabel="Reward"); ax.legend(); fig.tight_layout(); fig.savefig(f"{args.output_dir}/training_rewards.png", dpi=150); plt.close(fig)
    archive_path = save_report_assets(args.output_dir, report, logs, q_rewards, dqn_rewards)
    print("Data:", source); print("\\nTEST-SET COMPARISON\\n", report.to_string(index=False))
    print(f"\\nSaved results to {Path(args.output_dir).resolve()}")
    print(f"Downloadable archive: {archive_path}")

if __name__ == "__main__": main()
