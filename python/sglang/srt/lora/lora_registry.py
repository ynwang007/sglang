# Copyright 2023-2024 SGLang Team
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
# ==============================================================================


import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from uuid import uuid4

from sglang.srt.aio_rwlock import RWLock
from sglang.srt.utils import ConcurrentCounter


@dataclass
class LoRAInfo:
    """
    A data class to hold identification information for a LoRA model.

    This class encapsulates the name, path, and an optional unique identifier (ID) for a LoRA model. In the LoRA
    system, we use the LoRA ID to uniquely identify a LoRA model, which avoids potential conflicts caused by LoRA
    name reuses. In the future, we may also use it for correctly generating radix cache for a given LoRA.
    """

    lora_name: str
    lora_path: str
    lora_id: Optional[str] = field(default=None)

    def __post_init__(self):
        if self.lora_id is None:
            self.lora_id = uuid4().hex


class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters.

    This abstraction helps improve performance of LoRA state update while maintaining program correctness through essentially a simplified
    eventual consistency model between the "backend" model runner processes and the "frontend" tokenizer manager process: the LoRARegistry
    is hosted at the "frontend" main process (aka the tokenizer manager process) and serves as the sole source of truth for available LoRA
    adapters. Inference operations and dynamic LoRA updates can essentially be "overlapped" because the only two exclusive write operations
    ("register" and "unregister") are extremely lightweight. The actual time-consuming LoRA loading and unloading are done outside the
    critical areas and are asynchronously propagated to and executed at "backend" processes in a non-blocking manner.
    """

    def __init__(self, lora_paths: Optional[Dict[str, LoRAInfo]] = None):
        assert lora_paths is None or all(
            isinstance(lora, LoRAInfo) for lora in lora_paths.values()
        ), (
            "server_args.lora_paths should have been normalized to LoRAInfo objects during server initialization. "
            "Please file an issue if you see this error."
        )

        # A read-write lock to ensure thread-safe access to the registry.
        self._registry_lock = RWLock()
        # A dictionary to hold LoRAInfo objects, mapping from LoRA name to LoRAInfo.
        self._registry: Dict[str, LoRAInfo] = dict(lora_paths or {})
        # Counters for in-flight requests.
        self._counters: Dict[str, ConcurrentCounter] = defaultdict(ConcurrentCounter)

    async def register(self, lora_info: LoRAInfo):
        """
        Register a new LoRAInfo object in the registry.

        Args:
            lora_info (LoRAInfo): The LoRAInfo object to register.
        """
        async with self._registry_lock.writer_lock:
            if lora_info.lora_name in self._registry:
                raise ValueError(
                    f"LoRA with name {lora_info.lora_name} already exists. Loaded LoRAs: {self._registry.keys()}"
                )
            self._registry[lora_info.lora_name] = lora_info

    async def unregister(self, lora_name: str) -> str:
        """
        Unregister a LoRAInfo object from the registry and returns the removed LoRA ID.

        Args:
            lora_name (str): The name of the LoRA model to unregister.
        """
        async with self._registry_lock.writer_lock:
            lora_info = self._registry.get(lora_name, None)
            if lora_info is None:
                raise ValueError(
                    f"LoRA with name {lora_name} does not exist. Loaded LoRAs: {self._registry.keys()}"
                )
            del self._registry[lora_name]

        return lora_info.lora_id

    async def acquire(self, lora_name: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Queries registry for LoRA IDs based on LoRA names and start tracking the usage of the corresponding LoRA adapters
        by incrementing its counter.
        """

        async def _acquire_single(name: str) -> str:
            lora_info = self._registry.get(name, None)
            if lora_info is None:
                raise ValueError(
                    f"The following requested LoRA adapters are not loaded: {name}\n"
                    f"Loaded adapters: {self._registry.keys()}."
                )
            await self._counters[lora_info.lora_id].increment()
            return lora_info.lora_id

        async with self._registry_lock.reader_lock:
            if isinstance(lora_name, str):
                lora_id = await _acquire_single(lora_name)
                return lora_id
            elif isinstance(lora_name, list):
                lora_ids = await asyncio.gather(
                    *[_acquire_single(name) for name in lora_name]
                )
                return lora_ids
            else:
                raise TypeError(
                    "lora_name must be either a string or a list of strings."
                )

    async def release(self, lora_id: Union[str, List[str]]):
        """
        Decrements the usage counter for a LoRA adapter, indicating that it is no longer in use.
        """

        async with self._registry_lock.reader_lock:
            if isinstance(lora_id, str):
                await self._counters[lora_id].decrement()
            elif isinstance(lora_id, list):
                for id in lora_id:
                    await self._counters[id].decrement()
            else:
                raise TypeError("lora_id must be either a string or a list of strings.")

    async def wait_for_unload(self, lora_id: str):
        """
        Waits until the usage counter for a LoRA adapter reaches zero, indicating that it is no longer in use.
        This is useful for ensuring that a LoRA adapter can be safely unloaded.
        """
        assert (
            lora_id not in self._registry
        ), "wait_for_unload should only be called after the LoRA adapter has been unregistered. "
        counter = self._counters.get(lora_id)
        if counter:
            # Wait until no requests are using this LoRA adapter.
            await counter.wait_for_zero()
            del self._counters[lora_id]

    async def validate(self, loaded_adapters: Dict[str, LoRAInfo]):
        """
        The adapters registered in the `LoRARegistry` by design should always be a subset of the loaded adapters. This method
        validates this assertion to ensure program correctness.
        """
        async with self._registry_lock.reader_lock:
            assert self._registry.items() <= loaded_adapters.items(), (
                "Registry should always be a subset of loaded adapters. Please create a new issue if you see this error."
                f"Loaded adapters: {loaded_adapters}, registry: {self._registry}"
            )
