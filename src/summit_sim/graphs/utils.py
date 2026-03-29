"""Shared dependencies and application state container."""

from typing import Optional

from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.redis.aio import AsyncRedisStore


class AppState:
    """Thread-safe namespace for global application singletons.

    All attributes are initialized lazily in ApplicationLifecycle.setup_once()
    to ensure proper async event loop binding.
    """

    store: Optional[AsyncRedisStore] = None
    checkpointer: Optional[AsyncRedisSaver] = None
    author_graph: Optional[CompiledStateGraph] = None
    simulation_graph: Optional[CompiledStateGraph] = None
