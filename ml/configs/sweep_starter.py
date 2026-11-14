"""starter-v1: the 5 starter sweep configs (Step 4.1, plan Phase 4).

Exactly per plan:
    1. aggressive_shaping + ladder [1, 3],        cpu
    2. minimal_shaping   + ladder [1, 3, 6],      cpu
    3. sparse_curriculum + ladder [1, 3, 6, 9, 12], cpu
    4. aggressive_shaping + L=12 FIXED (curriculum ablation control), cpu
    5. aggressive_shaping + ladder [1, 3],        device=mps

3 seeds each -> 15 runs. M3: gamma/ent_coef axes are follow-up sweep ids,
NOT part of starter-v1 (reward/curriculum/device only).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

from ml.configs.base import RunConfig

STARTER_SWEEP_ID = "starter-v1"
STARTER_SEEDS: List[int] = [0, 1, 2]

STARTER_CONFIGS: Dict[str, RunConfig] = {
    "aggressive_shaping_ladder_1_3": RunConfig(
        name="aggressive_shaping_ladder_1_3",
        reward_config_name="aggressive_shaping",
        curriculum=[1, 3],
        device="cpu",
    ),
    "minimal_shaping_ladder_1_3_6": RunConfig(
        name="minimal_shaping_ladder_1_3_6",
        reward_config_name="minimal_shaping",
        curriculum=[1, 3, 6],
        device="cpu",
    ),
    "sparse_curriculum_ladder_1_3_6_9_12": RunConfig(
        name="sparse_curriculum_ladder_1_3_6_9_12",
        reward_config_name="sparse_curriculum",
        curriculum=[1, 3, 6, 9, 12],
        device="cpu",
    ),
    "aggressive_shaping_fixed_L12": RunConfig(
        name="aggressive_shaping_fixed_L12",
        reward_config_name="aggressive_shaping",
        curriculum=None,
        campaign_length=12,
        device="cpu",
    ),
    "aggressive_shaping_ladder_1_3_mps": RunConfig(
        name="aggressive_shaping_ladder_1_3_mps",
        reward_config_name="aggressive_shaping",
        curriculum=[1, 3],
        device="mps",
    ),
}


def select_configs(config_names: List[str]) -> Dict[str, RunConfig]:
    """Subset of STARTER_CONFIGS by name; raises on unknown names."""
    unknown = [name for name in config_names if name not in STARTER_CONFIGS]
    if unknown:
        raise SystemExit(
            f"unknown config name(s) {unknown}; "
            f"available: {sorted(STARTER_CONFIGS)}"
        )
    return {name: STARTER_CONFIGS[name] for name in config_names}


def config_with_seed(config: RunConfig, seed: int) -> RunConfig:
    """Per-run copy: only the seed differs from the named config."""
    return replace(config, seed=seed)


__all__ = [
    "STARTER_CONFIGS",
    "STARTER_SEEDS",
    "STARTER_SWEEP_ID",
    "config_with_seed",
    "select_configs",
]
