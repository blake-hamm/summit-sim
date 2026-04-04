"""Summit-Sim Chainlit web interface.

Entry point for the Chainlit application.
"""

import logging
from urllib.parse import parse_qs, urlparse

import chainlit as cl
import mlflow
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore
from redis.asyncio import Redis

from summit_sim.graphs.author import create_author_graph
from summit_sim.graphs.simulation import create_simulation_graph
from summit_sim.graphs.utils import AppState
from summit_sim.settings import settings
from summit_sim.ui import author, simulation

logger = logging.getLogger(__name__)


@cl.on_app_startup
async def on_app_startup() -> None:
    """Initialize global services once at server start."""
    # 1. MLflow first - agents depend on it for prompt registry
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    mlflow.pydantic_ai.autolog()  # type: ignore[attr-defined]
    logger.info("MLflow initialized")

    # 2. Redis persistence for LangGraph
    redis_client = Redis.from_url(settings.redis_url)
    AppState.redis_client = redis_client

    AppState.store = AsyncRedisStore(
        redis_client=redis_client,
        ttl={"default_ttl": 10080, "refresh_on_read": True},  # 7 Days
    )
    AppState.checkpointer = AsyncRedisSaver(
        redis_client=redis_client,
        ttl={"default_ttl": 1440, "refresh_on_read": True},  # 24 Hours
    )

    await AppState.store.setup()
    await AppState.checkpointer.setup()
    logger.info("Redis persistence initialized")

    # 3. Compile LangGraph graphs
    AppState.author_graph = create_author_graph(AppState.checkpointer, AppState.store)
    AppState.simulation_graph = create_simulation_graph(AppState.checkpointer)
    logger.info("Graphs compiled")

    # 4. Eagerly initialize agents (eliminates cold-start latency)
    from summit_sim.agents.utils import initialize_agents  # noqa: PLC0415

    initialize_agents()
    logger.info("All services initialized")


@cl.on_app_shutdown
async def on_app_shutdown() -> None:
    """Cleanup resources on server shutdown."""
    if AppState.redis_client:
        await AppState.redis_client.aclose()
        logger.info("Redis connection closed")


@cl.on_chat_start
async def start() -> None:
    """Initialize chat session - routes to author or player flow."""
    logger.info("New chat session started")
    query_string = ""
    environ = cl.context.session.environ if hasattr(cl.context, "session") else {}
    http_referer = environ.get("HTTP_REFERER", "")

    if http_referer:
        parsed = urlparse(http_referer)
        query_string = parsed.query

    params = parse_qs(query_string)
    scenario_id = params.get("scenario_id", [""])[0]

    # Check if scenario exists using async store
    if scenario_id and AppState.store is not None:
        scenario = await AppState.store.aget(("scenarios",), scenario_id)
        if scenario is not None:
            logger.info("Player joined session, scenario_id=%s", scenario_id)
            cl.user_session.set("mode", "player")
            cl.user_session.set("scenario_id", scenario_id)
            await simulation.start_simulation_session()
            return

    if scenario_id:
        logger.warning("Scenario not found, scenario_id=%s", scenario_id)
        await cl.Message(
            content=(
                "❌ Scenario not found. "
                "Please check the URL or ask your author for a valid link."
            ),
        ).send()
        return

    await ask_role_selection()


async def ask_role_selection() -> None:
    """Ask user to select their role."""
    element = cl.CustomElement(name="RoleSelection", props={})
    res = await cl.AskElementMessage(
        content=(
            "# 🏔️ Welcome to Summit-Sim\n\n"
            "**An AI-powered wilderness rescue simulator.**\n\n"
            "Summit-Sim generates curriculum-informed, interactive "
            "backcountry emergencies for dynamic Wilderness First "
            "Responder (WFR) training.\n\n"
            "---\n"
            "#### Select Your Role"
        ),
        element=element,
        timeout=settings.ui_timeout,
    ).send()

    if res and res.get("submitted"):
        role = res.get("role")
        if role not in ("instructor", "student"):
            await cl.Message(
                content="❌ Invalid role selected. Please try again.",
            ).send()
            await ask_role_selection()
            return

        cl.user_session.set("mode", role)

        if role == "instructor":
            await show_instructor_welcome()
        else:
            await show_student_welcome()

        logger.info("User selected role: %s", role)
        await author.ask_scenario_config()


async def show_instructor_welcome() -> None:
    """Display welcome message for instructors."""
    await cl.Message(
        content=(
            "### 🎓 Instructor Mode\n\n"
            "You can create scenarios, review them with full details visible, "
            "provide feedback to refine them, and share with students when ready."
        ),
    ).send()


async def show_student_welcome() -> None:
    """Display welcome message for students."""
    await cl.Message(
        content=(
            "### 👤 Student Mode\n\n"
            "Configure and play scenarios immediately. No hidden information "
            "will be shown - you'll need to discover medical details through "
            "assessment."
        ),
    ).send()


@cl.on_message
async def on_message_handler(_message: cl.Message) -> None:
    """Handle incoming messages (fallback handler)."""
    mode = cl.user_session.get("mode")

    if mode == "author":
        await cl.Message(
            content=(
                "🎨 Type **restart** to create a new scenario, "
                "or use the buttons above."
            ),
        ).send()
    elif mode == "player":
        await cl.Message(
            content="🎮 Please use the buttons above to make your selection.",
        ).send()
    else:
        await cl.Message(
            content="🎮 Welcome! The scenario creator will start automatically.",
        ).send()
        await start()
