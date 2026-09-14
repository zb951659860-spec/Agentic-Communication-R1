# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

# TrajectoryCollector and adjust_batch are imported lazily to avoid pulling in
# verl.utils.dataset (and its torchvision transitive dependency) at package
# import time.  Callers that need them should import directly:
#   from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
#   from agent_system.multi_turn_rollout.utils import adjust_batch

__all__ = []
