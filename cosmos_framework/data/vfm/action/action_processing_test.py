# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import torch

from cosmos_framework.data.vfm.action.action_processing import (
    StateActionSplitNormalizer,
    has_split_state_action_stats,
    resolve_action_normalization,
    resolve_state_action_normalizer,
)


def test_state_action_split_normalizer_applies_separate_stats():
    state_stats = {
        "q01": torch.zeros(14),
        "q99": torch.ones(14) * 2.0,
    }
    action_stats = {
        "q01": torch.full((14,), -1.0),
        "q99": torch.full((14,), 1.0),
    }
    normalizer = StateActionSplitNormalizer(
        state_normalizer=resolve_action_normalization("quantile", state_stats),
        action_normalizer=resolve_action_normalization("quantile", action_stats),
        use_state_row=True,
    )
    action = torch.tensor(
        [
            [1.0] * 14,
            [0.0] * 14,
        ],
        dtype=torch.float32,
    )
    normalized = normalizer.normalize_action(action)
    assert torch.allclose(normalized[0], torch.zeros(14))
    assert torch.allclose(normalized[1], torch.zeros(14))
    roundtrip = normalizer.denormalize_action(normalized)
    assert torch.allclose(roundtrip, action)


def test_resolve_state_action_normalizer_split_json(tmp_path):
    stats_path = tmp_path / "action_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "state": {"q01": [0.0] * 14, "q99": [2.0] * 14},
                "actions": {"q01": [-1.0] * 14, "q99": [1.0] * 14},
            }
        )
    )
    assert has_split_state_action_stats(json.loads(stats_path.read_text()))
    normalizer = resolve_state_action_normalizer("quantile", stats_path, use_state=True)
    assert isinstance(normalizer, StateActionSplitNormalizer)
