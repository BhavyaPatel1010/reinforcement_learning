"""A small, reproducible grocery inventory control environment.

Historical demand is replayed one day at a time.  The agent chooses a discrete
replenishment quantity; orders arrive after a configurable lead time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class InventoryConfig:
    lead_time: int = 2
    initial_inventory_days: float = 4.0
    max_inventory_days: float = 12.0
    # None scales the action quantities to the selected product's demand.
    order_step: int | None = None
    n_actions: int = 11
    holding_cost_rate: float = 0.015
    stockout_penalty_rate: float = 0.75
    unit_order_cost_rate: float = 0.55
    fixed_order_cost: float = 2.0
    sales_revenue_rate: float = 1.0


class InventoryEnv:
    """Discrete-action, finite-horizon environment without a Gym dependency."""

    def __init__(self, demand: Iterable[float], prices: Iterable[float] | None = None,
                 promotions: Iterable[float] | None = None, config: InventoryConfig | None = None,
                 reference_stats: dict | None = None):
        self.demand = np.maximum(np.asarray(list(demand), dtype=np.float32), 0)
        if len(self.demand) < 10:
            raise ValueError("At least 10 chronological demand observations are required.")
        self.prices = np.asarray(list(prices), dtype=np.float32) if prices is not None else np.ones(len(self.demand), dtype=np.float32)
        self.promotions = np.asarray(list(promotions), dtype=np.float32) if promotions is not None else np.zeros(len(self.demand), dtype=np.float32)
        if len(self.prices) != len(self.demand) or len(self.promotions) != len(self.demand):
            raise ValueError("demand, prices, and promotions must have the same length.")
        self.config = config or InventoryConfig()
        self.mean_demand = max(float(np.mean(self.demand)), 1.0)
        self.mean_price = max(float(np.mean(self.prices)), 0.01)
        if reference_stats is not None:
            self.mean_demand = reference_stats['mean_demand']
            self.mean_price = reference_stats['mean_price']
        self.reference_stats = dict(mean_demand=self.mean_demand, mean_price=self.mean_price)
        self.reward_scale = self.mean_demand * self.mean_price
        self.max_inventory = self.config.max_inventory_days * self.mean_demand
        # With 11 actions this makes the largest order roughly two days of mean
        # demand, whether the item sells 5 or 500 units per day.
        self.order_step = self.config.order_step or max(1, round(self.mean_demand / 5))
        self.action_values = np.arange(self.config.n_actions, dtype=np.float32) * self.order_step
        self.reset()

    @property
    def state_size(self) -> int:
        # inventory, pipeline, 3 demand lags, rolling mean, price, promotion, weekday
        return 9 + self.config.lead_time

    @property
    def action_size(self) -> int:
        return len(self.action_values)

    def reset(self, start_index: int = 0, horizon: int | None = None) -> np.ndarray:
        self.start_index = int(start_index)
        self.end_index = min(len(self.demand), self.start_index + (horizon or len(self.demand) - self.start_index))
        if self.end_index - self.start_index < 2:
            raise ValueError("Episode must contain at least two days.")
        self.t = self.start_index
        self.inventory = min(self.max_inventory, self.config.initial_inventory_days * self.mean_demand)
        self.pipeline = deque([0.0] * self.config.lead_time, maxlen=self.config.lead_time)
        self.history = deque([self.mean_demand] * 7, maxlen=7)
        return self._state()

    def _state(self) -> np.ndarray:
        i = min(self.t, len(self.demand) - 1)
        recent = list(self.history)
        weekday = (i % 7) / 6.0
        return np.array([
            self.inventory / self.max_inventory,
            sum(self.pipeline) / self.max_inventory,
            recent[-1] / self.mean_demand,
            recent[-2] / self.mean_demand,
            recent[-3] / self.mean_demand,
            np.mean(recent) / self.mean_demand,
            self.prices[i] / self.mean_price,
            self.promotions[i],
            weekday,
            *[quantity / self.max_inventory for quantity in self.pipeline],
        ], dtype=np.float32)

    def step(self, action: int):
        if not 0 <= int(action) < self.action_size:
            raise ValueError("Invalid action index")
        arriving = self.pipeline.popleft() if self.config.lead_time else 0.0
        self.inventory = min(self.max_inventory, self.inventory + arriving)
        order_qty = float(self.action_values[int(action)])
        if self.config.lead_time:
            self.pipeline.append(order_qty)
        else:
            self.inventory = min(self.max_inventory, self.inventory + order_qty)

        demand = float(self.demand[self.t])
        price = float(max(self.prices[self.t], 0.01))
        sales = min(self.inventory, demand)
        lost_sales = demand - sales
        self.inventory -= sales
        self.history.append(demand)

        sales_revenue = sales * price * self.config.sales_revenue_rate
        order_cost = order_qty * price * self.config.unit_order_cost_rate + (self.config.fixed_order_cost if order_qty else 0.0)
        holding_cost = self.inventory * price * self.config.holding_cost_rate
        stockout_cost = lost_sales * price * self.config.stockout_penalty_rate
        profit = sales_revenue - order_cost - holding_cost - stockout_cost
        # Normalization keeps DQN targets numerically stable across cheap and
        # expensive products while preserving the optimal policy.
        reward = profit / self.reward_scale
        info = {"demand": demand, "sales": sales, "lost_sales": lost_sales, "order_qty": order_qty,
                "inventory": self.inventory, "sales_revenue": sales_revenue, "order_cost": order_cost,
                "holding_cost": holding_cost, "stockout_cost": stockout_cost, "profit": profit}
        self.t += 1
        done = self.t >= self.end_index
        return self._state(), float(reward), done, info
