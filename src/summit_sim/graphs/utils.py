"""Shared dependencies and application state container."""

from typing import TYPE_CHECKING

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.redis.aio import AsyncRedisStore
from redis.asyncio import Redis

if TYPE_CHECKING:
    pass


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
