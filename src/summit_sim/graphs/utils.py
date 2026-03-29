"""Shared utilities for LangGraph state definitions."""

from langgraph.store.redis.aio import AsyncRedisStore

from summit_sim.settings import settings


class _ScenarioStore:
    """Lazy singleton for scenario store."""

    _instance: AsyncRedisStore | None = None

    @classmethod
    async def get(cls) -> AsyncRedisStore:
        """Get or create the scenario store singleton.

        TTL: 7 days (10080 minutes) for scenarios.

        Returns:
            Initialized AsyncRedisStore instance.

        """
        if cls._instance is None:
            cls._instance = AsyncRedisStore(
                settings.redis_url,
                ttl={"default_ttl": 10080, "refresh_on_read": True},
            )
            await cls._instance.setup()
        return cls._instance


async def get_scenario_store() -> AsyncRedisStore:
    """Get the global scenario store instance.

    Initializes on first call after event loop is running.

    Returns:
        Initialized AsyncRedisStore instance.

    """
    return await _ScenarioStore.get()
