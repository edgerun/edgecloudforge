"""
Copyright 2024 b<>com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
from typing import Set, Tuple, TYPE_CHECKING, Dict, List

import xgboost
from simpy import Environment, Store

from src.policy.knative.model import KnativeSchedulerState, KnativeSystemState
from src.policy.proactiveknative.model import ProactiveKnativeSystemState
from src.train import load_models
from src.policy.knative.util import count_events_in_windows_ts
from src.motivational.constants import PREDICTION_WINDOW_SIZE

if TYPE_CHECKING:
    from src.placement.infrastructure import Node, Platform, Task

from src.placement.model import (
    DurationSecond,
    PlatformVector,
    SchedulerState,
    SizeGigabyte,
    SpeedMBps,
    SystemState,
    TaskType, SimulationData, SimulationPolicy,
)

from src.placement.autoscaler import Autoscaler

logger = logging.getLogger(__name__)


class BaasKnativeAutoscaler(Autoscaler):

    def __init__(
            self,
            env: Environment,
            mutex: Store,
            data: SimulationData,
            policy: SimulationPolicy,
            resource_prediction_model: xgboost.XGBRegressor
    ):
        super().__init__(env, mutex, data, policy)
        self.resource_prediction_model = resource_prediction_model
        self.container_predictions = []
        self.n_bb = 0
        self.t = 0

    def burst_aware_autoscaling(self, system_state: ProactiveKnativeSystemState, task_type: TaskType, k=10):
        """
        Implementation of the burst-aware autoscaling algorithm

        Parameters:
        -----------
        response_time_slo: float
            Response time SLO threshold
        resource_prediction_model: object
            Trained resource prediction model
        k: int
            Window size for workload forecasting and burst detection

        Returns:
        --------
        dict
            Dictionary containing scaling decisions and burst detection results
        """
        results = {
            'time_intervals': [],
            'workload': [],
            'predicted_workload': [],
            'container_count': [],
            'is_burst': [],
            'sigma_max': []
        }

        is_burst = False

        # Simulate time intervals

        # Forecast workload for next interval using Elastic Net
        look_forward_size = PREDICTION_WINDOW_SIZE
        events = \
            count_events_in_windows_ts(self.env.now, system_state.time_series, task_type['name'], look_forward_size,
                                       look_forward_size)
        if events is None:
            # print(f'FN {task_type["name"]} has no events')
            return {"any": 0}
        events = events[0] / look_forward_size

        # Predict required containers using resource prediction model
        # For simplicity, we'll use a fixed response time
        # TODO we assume that maxDurationDeviation is 15 every time
        response_time_slo = max(task_type["executionTime"].values()) * 1000 * 15
        predicted_containers = self.resource_prediction_model.predict([[events, response_time_slo]])[0]
        print(predicted_containers)
        # Store prediction
        container_predictions = self.container_predictions
        container_predictions.append(predicted_containers)

        # Burst detection logic
        sigma_max = 0
        n_max = 0

        # We need at least k+1 predictions for burst detection
        if len(container_predictions) > k:
            # Calculate standard deviation for different window sizes
            for i in range(1, k + 1):
                if self.t - i >= 0:
                    # Get the last i+1 predictions (including current)
                    window = container_predictions[-i - 1:]
                    sigma = np.std(window)

                    if sigma > sigma_max:
                        sigma_max = sigma
                        n_max = max(window)

            # Determine if burst and set container count
            if sigma_max >= 2 and not is_burst:
                container_count = n_max
                is_burst = True
                self.n_bb = container_predictions[-2]  # Store container count before burst
            elif sigma_max >= 2 and is_burst:
                container_count = n_max
            elif sigma_max < 2 and is_burst:
                if self.n_bb > predicted_containers:
                    container_count = predicted_containers
                    is_burst = False
                    self.n_bb = 0
                else:
                    container_count = n_max
            else:  # sigma_max < 2 and not is_burst
                container_count = predicted_containers
        else:
            # Not enough history for burst detection
            container_count = predicted_containers
            sigma_max = 0

        # Store results
        results['predicted_workload'].append(events)
        results['container_count'].append(container_count)
        results['is_burst'].append(is_burst)
        results['sigma_max'].append(sigma_max)
        current_replicas = len(system_state.replicas[task_type['name']])

        modify_replicas = container_count - current_replicas
        return {'any': modify_replicas}

    def scaling_level(self, system_state: ProactiveKnativeSystemState, task_type: TaskType):
        # Scheduling functions called in a Simpy Process must be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield
        result = self.burst_aware_autoscaling(system_state, task_type)
        self.t += 1
        return result

    def create_first_replica(self, system_state: SystemState, task_type: TaskType):
        # Knative will allocate a new CPU replica
        available_hardware: Set[str] = set()
        for _, platforms in system_state.available_resources.items():
            for platform in platforms:
                if (
                        # platform.type["hardware"] == "cpu"
                        # and platform.type["shortName"] in task_type["platforms"]
                        platform.type["shortName"]
                        in task_type["platforms"]
                ):
                    available_hardware.add(platform.type["shortName"])

        stop = None
        # FIXME: What if no available hardware?
        for platform_name in available_hardware:
            stop = yield self.env.process(
                self.scale_up(
                    1,
                    system_state,
                    task_type["name"],
                    self.data.platform_types[platform_name]["shortName"],
                )
            )

            if not isinstance(stop, StopIteration):
                # Resource found, stop iterating
                break

        return stop

    def create_replica(
            self, couples_suitable: Set[Tuple[Node, Platform]], task_type: TaskType
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489
        if False:
            yield

        """
        # Knative only allocates CPUs
        filtered_couples = set(filter(
            lambda couple: couple[1].type["hardware"] == "cpu",
            couples_suitable
        ))
        """

        # Knative selects a replica on the most available node (cf. ENSURE)
        available_couple = max(
            # filtered_couples, key=lambda couple: couple[0].available_platforms
            couples_suitable,
            key=lambda couple: couple[0].available_platforms,
        )

        return available_couple

    def initialize_replica(
            self,
            new_replica: Tuple[Node, Platform],
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            system_state: KnativeSystemState,
    ):
        node: Node = new_replica[0]
        platform: Platform = new_replica[1]

        # Check node RAM cache
        warm_function: bool = (
                platform.previous_task is not None
                and platform.previous_task.type["name"] == task_type["name"]
        )

        # Initialize image retrieval duration
        retrieval_duration: DurationSecond = 0.0

        # TODO: Retrieve image if function not in RAM cache nor in disk cache
        # FIXME: Should be factored in superclass
        if not warm_function:
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )

            # Update image retrieval duration
            retrieval_size: SizeGigabyte = task_type["imageSize"][
                platform.type["shortName"]
            ]
            # Depends on storage performance
            # FIXME: What's the policy for storage selection?
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            # Depends on network link speed
            retrieval_speed: SpeedMBps = min(
                node_storage.type["throughput"]["write"], node.network["bandwidth"]
            )
            retrieval_duration += (
                    retrieval_size / (retrieval_speed / 1024)
                    + node_storage.type["latency"]["write"]
            )

            # print(f"retrieval size = {retrieval_size}")
            # print(f"retrieval speed = {retrieval_speed}")

            # TODO: Update disk usage
            stored = node_storage.store_function(platform.type["shortName"], task_type)

            if not stored:
                logging.error(
                    f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                    f" cache image for {self}"
                )

            # Release storage
            yield node.storage.put(node_storage)

        # print(f"retrieval duration = {retrieval_duration}")

        # Update state
        # FIXME: Move to state update methods
        state: KnativeSchedulerState = system_state.scheduler_state
        # Knative policy
        state.average_contention[task_type["name"]][
            (new_replica[0].id, new_replica[1].id)
        ] = 1.0

        # FIXME: Retrieve function image
        yield self.env.timeout(retrieval_duration)

        # FIXME: Update platform time spent on storage
        platform.storage_time += retrieval_duration

        # FIXME: Double initialize bug...
        try:
            # Set platform to ready state
            yield platform.initialized.succeed()
        except RuntimeError:
            """
            logging.error(
                f"[ {self.env.now} ] Autoscaler tried to initialize "
                f"{new_replica[1]} ({new_replica[0]}) but it was already initialized."
            )

            logging.error(
                f"[ {self.env.now} ] Last allocation time: "
                f"{new_replica[1].last_allocated} "
                " -- Last removal time: "
                f"{new_replica[1].last_removed}"
            )
            """
            pass

        # Statistics (Node)
        node.cache_hits += 0

    def remove_replica(
            self,
            function_replicas: Set[Tuple[Node, Platform]],
            task_type: TaskType,
            system_state: KnativeSystemState,
    ):
        # Scaling functions that do not yield values must still be Generators
        # No-op as per https://stackoverflow.com/a/68628599/9568489&
        if False:
            yield

        # Sort function replicas by in-flight requests count
        sorted_replicas = sorted(
            function_replicas, key=lambda couple: len(couple[1].queue.items)
        )

        # Mark replica for removal if its task queue is empty
        # Return None if no replica can be removed
        removed_couple = next(
            (
                replica
                for replica in sorted_replicas
                if not replica[1].queue.items
                   and not replica[1].current_task
                   and (self.env.now - replica[1].idle_since) > self.policy.keep_alive
            ),
            None,
        )

        if removed_couple:
            # Update state
            # FIXME: Move to state update methods
            state: SchedulerState = system_state.scheduler_state
            try:
                # Knative policy
                del state.average_contention[task_type["name"]][
                    (removed_couple[0].id, removed_couple[1].id)
                ]
            except KeyError:
                """
                logging.error(
                    f"[ {self.env.now} ] Autoscaler tried to scale down "
                    f"{task_type['name']}, but {removed_couple[1]} was already removed"
                )
                """
                pass

        return removed_couple
