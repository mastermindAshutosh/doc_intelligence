import pytest
import asyncio
import torch
from unittest.mock import MagicMock
from backend.serving.batcher import DynamicBatcher
from backend.serving.service import ClassificationService
from backend.schemas import ClassificationResponse

class MockModel:
    def forward_inference(self, feats_batch):
        return torch.zeros(feats_batch.shape[0], 5), 3

@pytest.fixture
def batcher():
    b = DynamicBatcher(model=MockModel(), max_batch=2, flush_ms=20)
    return b

@pytest.mark.asyncio
async def test_dynamic_batcher_collects_batch(batcher):
    # Fire up background task
    task = asyncio.create_task(batcher.run())
    
    # Send 2 fast requests
    t1 = asyncio.create_task(batcher.infer(torch.zeros(256)))
    t2 = asyncio.create_task(batcher.infer(torch.zeros(256)))
    
    res1, res2 = await asyncio.gather(t1, t2)
    
    assert res1.shape == (5,)
    assert res2.shape == (5,)
    assert batcher.last_exit_layer == 3
    task.cancel()

@pytest.mark.asyncio
async def test_batcher_flushes_on_timeout(batcher):
    task = asyncio.create_task(batcher.run())
    
    # Send 1 request, batch size is 2 so it must hit timeout
    res = await batcher.infer(torch.zeros(256))
    
    assert res.shape == (5,)
    task.cancel()

@pytest.mark.asyncio
async def test_classify_returns_correct_schema():
    # Setup mock service
    class MockIngester:
        def ingest(self, raw): return MagicMock()
        
    class MockBuilder:
        def build(self, ing): return MagicMock()
        
    class MockOOD:
        def is_ood(self, log, emb): return False
        
    class MockScaler:
        def scale(self, log): 
            t = torch.zeros(1, 5)
            t[0, 0] = 1.0 # 100% conf
            return t
            
    class MockRouter:
        def route(self, p, c, o): return "direct"
        
    svc = ClassificationService(
        MockIngester(), MockBuilder(), None, None, None,
        DynamicBatcher(model=MockModel(), max_batch=1, flush_ms=10),
        MockOOD(), MockScaler(), MockRouter()
    )
    
    task = asyncio.create_task(svc.batcher.run())
    
    resp = await svc.classify(b"dummy")
    
    assert isinstance(resp, ClassificationResponse)
    assert resp.prediction == "TAX"
    assert resp.confidence == 1.0
    assert resp.routing == "direct"
    assert resp.ood_fired is False
    
    task.cancel()

@pytest.mark.asyncio
async def test_ood_flag_in_response():
    # Same as above but OOD returns True
    class MockOOD:
        def is_ood(self, log, emb): return True
        
    class MockScaler:
        def scale(self, log): return torch.zeros(1, 5)
        
    class MockRouter:
        def route(self, p, c, o): return "ood"
        
    svc = ClassificationService(
        MagicMock(), MagicMock(), None, None, None,
        DynamicBatcher(model=MockModel(), max_batch=1, flush_ms=10),
        MockOOD(), MockScaler(), MockRouter()
    )
    
    task = asyncio.create_task(svc.batcher.run())
    resp = await svc.classify(b"dummy code bytes")
    
    assert resp.prediction == "__UNCERTAIN__"
    assert resp.routing == "ood"
    assert resp.ood_fired is True
    
    task.cancel()

@pytest.mark.asyncio
async def test_audit_fires_without_blocking():
    svc = ClassificationService(
        MagicMock(), MagicMock(), None, None, None,
        DynamicBatcher(model=MockModel(), max_batch=1, flush_ms=10),
        MagicMock(), MagicMock(), MagicMock()
    )
    svc.ood_ensemble.is_ood.return_value = False
    svc.temp_scaler.scale.return_value = torch.zeros(1, 5)
    svc.router.route.return_value = "human_review"
    
    task = asyncio.create_task(svc.batcher.run())
    
    resp = await svc.classify(b"bytes bytes")
    
    # Wait a tiny bit for the audit task to fire
    await asyncio.sleep(0.01)
    
    assert len(svc.audit_logs) == 1
    task.cancel()

@pytest.mark.asyncio
async def test_cache_hit_skips_ingestion():
    svc = ClassificationService(
        MagicMock(), MagicMock(), None, None, None,
        DynamicBatcher(model=MockModel(), max_batch=1, flush_ms=10),
        MagicMock(), MagicMock(), MagicMock()
    )
    svc.ood_ensemble.is_ood.return_value = False
    svc.temp_scaler.scale.return_value = torch.zeros(1, 5)
    svc.router.route.return_value = "direct"
    
    # Inject a cached item
    import hashlib
    h = hashlib.sha256(b"hit cache").hexdigest()
    svc.cache[h] = (torch.zeros(256), torch.zeros(256))
    
    task = asyncio.create_task(svc.batcher.run())
    
    resp = await svc.classify(b"hit cache")
    assert not svc.ingester.ingest.called
    
    task.cancel()
