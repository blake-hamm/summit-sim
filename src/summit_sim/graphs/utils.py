"""Shared dependencies and application state container."""

from typing import TYPE_CHECKING

from langgraph._internal._retry import default_retry_on
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.redis.aio import AsyncRedisStore
from langgraph.types import RetryPolicy
from pydantic_ai.exceptions import ModelAPIError
from redis.asyncio import Redis

if TYPE_CHECKING:
    pass


def should_retry_llm_errors(exc: Exception) -> bool:
    """Retry on transient LLM errors (timeouts, connection issues).

    ModelAPIError wraps timeout and connection errors from the LLM provider.
    LangGraph's default excludes RuntimeError (which ModelAPIError inherits from),
    so we explicitly retry on ModelAPIError.
    """
    if isinstance(exc, ModelAPIError):
        return True
    return default_retry_on(exc)


retry_policy = RetryPolicy(
    max_attempts=3,
    retry_on=should_retry_llm_errors,
)


class AppState:
    """Thread-safe namespace for global application singletons.

    All attributes are initialized eagerly in on_app_startup()
    to eliminate cold-start latency and race conditions.
    """

    redis_client: Redis | None = None
    store: AsyncRedisStore | None = None
    checkpointer: AsyncRedisSaver | None = None
    author_graph: CompiledStateGraph | None = None
    simulation_graph: CompiledStateGraph | None = None
