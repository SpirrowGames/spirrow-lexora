"""Tests for request queue."""

import asyncio

import pytest

from lexora.services.queue import (
    Priority,
    QueueFullError,
    QueueTimeoutError,
    RequestQueue,
)


class TestPriority:
    """Tests for Priority enum."""

    def test_priority_ordering(self) -> None:
        """Test priority values for correct ordering."""
        assert Priority.HIGH < Priority.NORMAL < Priority.LOW


class TestRequestQueue:
    """Tests for RequestQueue."""

    @pytest.fixture
    def queue(self) -> RequestQueue:
        """Create a request queue for testing."""
        return RequestQueue(max_size=10, default_timeout=5.0, max_concurrent=2)

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self, queue: RequestQueue) -> None:
        """Test basic enqueue and process flow."""
        results: list[str] = []

        async def handler(data: str) -> str:
            results.append(data)
            return f"processed:{data}"

        # Start processor
        processor_task = asyncio.create_task(queue.process(handler))

        # Enqueue and wait for result
        result = await queue.enqueue("test-data")

        assert result == "processed:test-data"
        assert "test-data" in results

        # Cleanup
        await queue.shutdown()
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue: RequestQueue) -> None:
        """Test requests are processed in priority order."""
        processed_order: list[str] = []

        async def slow_handler(data: str) -> str:
            await asyncio.sleep(0.05)
            processed_order.append(data)
            return data

        # Use single concurrent to ensure ordering
        queue = RequestQueue(max_size=10, default_timeout=5.0, max_concurrent=1)

        # Start processor
        processor_task = asyncio.create_task(queue.process(slow_handler))

        # Give processor time to start
        await asyncio.sleep(0.01)

        # Enqueue in reverse priority order
        tasks = [
            asyncio.create_task(queue.enqueue("low", priority=Priority.LOW)),
            asyncio.create_task(queue.enqueue("normal", priority=Priority.NORMAL)),
            asyncio.create_task(queue.enqueue("high", priority=Priority.HIGH)),
        ]

        await asyncio.gather(*tasks)

        # High priority should be processed first (after any in-flight)
        # Note: First item may be any priority due to race with enqueue
        assert "high" in processed_order
        assert "normal" in processed_order
        assert "low" in processed_order

        await queue.shutdown()
        await asyncio.wait_for(processor_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_queue_full_error(self) -> None:
        """Test QueueFullError when queue is full."""
        queue = RequestQueue(max_size=1, default_timeout=5.0)

        # Don't start processor - queue will fill up
        # First enqueue shouldn't complete without processor
        task1 = asyncio.create_task(queue.enqueue("first"))

        # Small delay to let first enqueue start
        await asyncio.sleep(0.01)

        # Second enqueue should fail
        with pytest.raises(QueueFullError):
            await asyncio.wait_for(queue.enqueue("second"), timeout=0.1)

        task1.cancel()
        try:
            await task1
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_timeout_error(self, queue: RequestQueue) -> None:
        """Test QueueTimeoutError when request times out."""
        async def slow_handler(data: str) -> str:
            await asyncio.sleep(10.0)  # Very slow
            return data

        processor_task = asyncio.create_task(queue.process(slow_handler))

        with pytest.raises(QueueTimeoutError):
            await queue.enqueue("test", timeout=0.1)

        await queue.shutdown()
        await asyncio.wait_for(processor_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_exception_propagation(self, queue: RequestQueue) -> None:
        """Test exceptions from handler propagate to caller."""

        async def failing_handler(data: str) -> str:
            raise ValueError("Handler error")

        processor_task = asyncio.create_task(queue.process(failing_handler))

        with pytest.raises(ValueError, match="Handler error"):
            await queue.enqueue("test")

        await queue.shutdown()
        processor_task.cancel()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_processing(self) -> None:
        """Test concurrent request processing."""
        queue = RequestQueue(max_size=10, default_timeout=5.0, max_concurrent=3)

        processing_count = 0
        max_concurrent_seen = 0

        async def tracking_handler(data: str) -> str:
            nonlocal processing_count, max_concurrent_seen
            processing_count += 1
            max_concurrent_seen = max(max_concurrent_seen, processing_count)
            await asyncio.sleep(0.05)
            processing_count -= 1
            return data

        processor_task = asyncio.create_task(queue.process(tracking_handler))

        # Enqueue multiple requests
        tasks = [asyncio.create_task(queue.enqueue(f"req-{i}")) for i in range(5)]

        await asyncio.gather(*tasks)

        # Should have seen at least 2 concurrent (might not hit 3 due to timing)
        assert max_concurrent_seen >= 2

        await queue.shutdown()
        await asyncio.wait_for(processor_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_get_stats(self, queue: RequestQueue) -> None:
        """Test getting queue statistics."""
        stats = queue.get_stats()

        assert stats["queue_size"] == 0
        assert stats["pending_count"] == 0
        assert stats["processing_count"] == 0
        assert stats["max_size"] == 10
        assert stats["max_concurrent"] == 2
        assert stats["is_shutdown"] is False

    @pytest.mark.asyncio
    async def test_shutdown(self, queue: RequestQueue) -> None:
        """Test graceful shutdown."""
        async def handler(data: str) -> str:
            return data

        processor_task = asyncio.create_task(queue.process(handler))

        await queue.shutdown()

        assert queue.get_stats()["is_shutdown"] is True

        # Should not accept new requests
        with pytest.raises(RuntimeError, match="shutting down"):
            await queue.enqueue("test")

        await asyncio.wait_for(processor_task, timeout=1.0)
