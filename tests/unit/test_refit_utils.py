# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from molt.trainer.fsdp.refit import should_refit_state_dict_entry, state_dict_parameter_trainability


class _TiedHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 3)
        self.lm_head = torch.nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


def test_state_dict_trainability_preserves_tied_parameter_aliases():
    model = _TiedHeadModel()

    trainability = state_dict_parameter_trainability(model)

    assert "embed_tokens.weight" in trainability
    assert "lm_head.weight" in trainability
    assert should_refit_state_dict_entry(
        "lm_head.weight",
        model.state_dict()["lm_head.weight"],
        trainability,
        is_vlm=False,
    )


def test_refit_filter_skips_non_parameter_buffers():
    trainability = {}

    assert not should_refit_state_dict_entry(
        "layers.0.mlp.gate.e_score_correction_bias", torch.zeros(2), trainability, is_vlm=False
    )
    assert not should_refit_state_dict_entry("some_metadata", torch.zeros(2), trainability, is_vlm=False)
