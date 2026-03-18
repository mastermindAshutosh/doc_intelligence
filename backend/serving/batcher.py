import asyncio
import torch
from torch import Tensor
from backend.classification.model import MultiExitClassifier
from backend.config import settings

class DynamicBatcher:
    def __init__(self, model: MultiExitClassifier, max_batch: int = 32, flush_ms: int = 20):
        self.model     = model
        self.max_batch = max_batch
        self.flush_ms  = flush_ms
        self._queue    = asyncio.Queue()
        self._flush    = asyncio.Event()
        self.last_exit_layer = 12

    async def infer(self, features: Tensor) -> Tensor:
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._queue.put((features, fut))
        if self._queue.qsize() >= self.max_batch:
            self._flush.set()
        return await fut

    async def run(self):
        """Start this as an asyncio background task on app startup."""
        while True:
            try:
                # Wait for max_batch size, or timeout flush_ms
                await asyncio.wait_for(self._flush.wait(), timeout=self.flush_ms / 1000)
            except asyncio.TimeoutError:
                pass
            
            self._flush.clear()
            items = []
            while not self._queue.empty() and len(items) < self.max_batch:
                items.append(self._queue.get_nowait())
            
            if not items:
                continue
            
            feats_batch = torch.stack([f for f, _ in items])
            with torch.no_grad():
                logits_batch, exit_layer = self.model.forward_inference(feats_batch)
            
            self.last_exit_layer = exit_layer
            
            for i, (_, fut) in enumerate(items):
                if not fut.done():
                    fut.set_result(logits_batch[i])
