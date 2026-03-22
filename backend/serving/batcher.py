import asyncio
import torch
from torch import Tensor
from backend.classification.model import MultiExitClassifier


class DynamicBatcher:
    """
    Collects inference requests into batches and runs them through the
    classifier in one forward pass. Results (logits + exit_layer) are
    returned per-item via asyncio futures.

    Key fixes vs original:
    - exit_layer is now returned through the future alongside logits,
      eliminating the shared-mutable-attribute race condition where
      service.py could read a stale last_exit_layer from a concurrent batch.
    - asyncio.get_event_loop() replaced with asyncio.get_running_loop()
      (the former is deprecated in Python 3.10+ and raises a DeprecationWarning
      in async contexts).
    """

    def __init__(
        self,
        model: MultiExitClassifier,
        max_batch: int = 32,
        flush_ms: int = 20,
    ):
        self.model     = model
        self.max_batch = max_batch
        self.flush_ms  = flush_ms
        self._queue    = asyncio.Queue()
        self._flush    = asyncio.Event()

    async def infer(self, features: Tensor) -> tuple[Tensor, int]:
        """
        Submit a single feature tensor for batched inference.

        Returns:
            logits:     (n_classes,) tensor for this item
            exit_layer: which layer this item's batch exited at
        """
        fut = asyncio.get_running_loop().create_future()
        await self._queue.put((features, fut))

        # Signal the worker to flush early if we've hit max_batch
        if self._queue.qsize() >= self.max_batch:
            self._flush.set()

        # Result is (logits_i, exit_layer) — unpacked by service.py
        return await fut

    async def run(self):
        """
        Background worker task. Started in app lifespan, cancelled on shutdown.

        Waits for flush_ms or max_batch, then drains the queue and runs
        one batched forward pass. Each item's future receives its own
        (logits, exit_layer) tuple — no shared state between requests.
        """
        while True:
            try:
                await asyncio.wait_for(
                    self._flush.wait(),
                    timeout=self.flush_ms / 1000,
                )
            except asyncio.TimeoutError:
                pass

            self._flush.clear()

            # Drain up to max_batch items
            items: list[tuple[Tensor, asyncio.Future]] = []
            while not self._queue.empty() and len(items) < self.max_batch:
                items.append(self._queue.get_nowait())

            if not items:
                continue

            feats_batch = torch.stack([f for f, _ in items])   # (B, D)

            with torch.no_grad():
                logits_batch, exit_layer = self.model.forward_inference(feats_batch)

            # Resolve each future with (logits_i, exit_layer)
            # exit_layer is the same for all items in this batch — it reflects
            # the deepest layer any item in the batch needed.
            for i, (_, fut) in enumerate(items):
                if not fut.done():
                    fut.set_result((logits_batch[i], exit_layer))
