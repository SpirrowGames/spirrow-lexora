"""Priority queue for request management."""

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Generic, TypeVar
from uuid import uuid4

from lexora.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Priority(IntEnum):
    """Request priority levels.

    Lower values = higher priority.
    """

    HIGH = 0
    NORMAL = 10
    LOW = 20


@dataclass(order=True)
class QueueItem(Generic[T]):
    """Item in the priority queue.

    Args:
        priority: Priority level (lower = higher priority).
        timestamp: Creation timestamp for FIFO ordering within same priority.
        request_id: Unique request identifier.
        data: The actual request data.
        future: Future to set when the request is processed.
    """

    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    data: T = field(compare=False)
    future: asyncio.Future[Any] = field(compare=False)


class QueueFullError(Exception):
    """Raised when the queue is full."""

    pass


class QueueTimeoutError(Exception):
    """Raised when a request times out waiting in queue."""

    pass


class RequestQueue:
    """Priority queue for managing LLM requests.

    Provides priority-based request queuing with configurable concurrency
    and timeout handling.

    Args:
        max_size: Maximum number of items in the queue.
        default_timeout: Default timeout for requests in seconds.
        max_concurrent: Maximum concurrent requests being processed.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_timeout: float = 60.0,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the request queue.

        Args:
            max_size: Maximum number of items in the queue.
            default_timeout: Default timeout for requests in seconds.
            max_concurrent: Maximum concurrent requests being processed.
        """
        self.max_size = max_size
        self.default_timeout = default_timeout
        self.max_concurrent = max_concurrent

        self._queue: asyncio.PriorityQueue[QueueItem[Any]] = asyncio.PriorityQueue(
            maxsize=max_size
        )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending: dict[str, QueueItem[Any]] = {}
        self._processing: set[str] = set()
        self._shutdown = False

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def pending_count(self) -> int:
        """Get number of pending requests."""
        return len(self._pending)

    @property
    def processing_count(self) -> int:
        """Get number of requests currently being processed."""
        return len(self._processing)

    async def enqueue(
        self,
        data: T,
        priority: Priority | int = Priority.NORMAL,
        timeout: float | None = None,
    ) -> Any:
        """Add a request to the queue and wait for result.

        Args:
            data: Request data.
            priority: Request priority.
            timeout: Request timeout in seconds. Uses default if None.

        Returns:
            Result from processing the request.

        Raises:
            QueueFullError: If the queue is full.
            QueueTimeoutError: If the request times out.
            Exception: Any exception raised during processing.
        """
        if self._shutdown:
            raise RuntimeError("Queue is shutting down")

        if self._queue.full():
            raise QueueFullError("Request queue is full")

        request_id = str(uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()

        item = QueueItem(
            priority=int(priority),
            timestamp=loop.time(),
            request_id=request_id,
            data=data,
            future=future,
        )

        self._pending[request_id] = item

        try:
            await self._queue.put(item)
            logger.debug(
                "request_enqueued",
                request_id=request_id,
                priority=priority,
                queue_size=self.size,
            )

            effective_timeout = timeout if timeout is not None else self.default_timeout
            return await asyncio.wait_for(future, timeout=effective_timeout)

        except asyncio.TimeoutError:
            logger.warning(
                "request_timeout",
                request_id=request_id,
                timeout=timeout or self.default_timeout,
            )
            raise QueueTimeoutError(
                f"Request {request_id} timed out after {timeout or self.default_timeout}s"
            )

        finally:
            self._pending.pop(request_id, None)

    async def process(
        self,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> None:
        """Process requests from the queue continuously.

        This is the worker loop that processes queued requests.

        Args:
            handler: Async function to process each request.
        """
        logger.info("queue_worker_started")

        while not self._shutdown:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Check if request was already cancelled/timed out
            if item.future.done():
                self._queue.task_done()
                continue

            # Process with concurrency limit
            asyncio.create_task(self._process_item(item, handler))

        logger.info("queue_worker_stopped")

    async def _process_item(
        self,
        item: QueueItem[Any],
        handler: Callable[[Any], Awaitable[Any]],
    ) -> None:
        """Process a single queue item.

        Args:
            item: Queue item to process.
            handler: Handler function.
        """
        async with self._semaphore:
            self._processing.add(item.request_id)

            try:
                if item.future.done():
                    return

                logger.debug(
                    "request_processing",
                    request_id=item.request_id,
                    priority=item.priority,
                )

                result = await handler(item.data)

                if not item.future.done():
                    item.future.set_result(result)

                logger.debug(
                    "request_completed",
                    request_id=item.request_id,
                )

            except Exception as e:
                if not item.future.done():
                    item.future.set_exception(e)

                logger.error(
                    "request_failed",
                    request_id=item.request_id,
                    error=str(e),
                )

            finally:
                self._processing.discard(item.request_id)
                self._queue.task_done()

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Gracefully shut down the queue.

        Args:
            timeout: Maximum time to wait for pending requests.
        """
        logger.info("queue_shutdown_started")
        self._shutdown = True

        # Cancel all pending futures
        for item in list(self._pending.values()):
            if not item.future.done():
                item.future.cancel()

        # Wait for processing to complete
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("queue_shutdown_timeout", timeout=timeout)

        logger.info("queue_shutdown_complete")

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics.
        """
        return {
            "queue_size": self.size,
            "pending_count": self.pending_count,
            "processing_count": self.processing_count,
            "max_size": self.max_size,
            "max_concurrent": self.max_concurrent,
            "is_shutdown": self._shutdown,
        }
