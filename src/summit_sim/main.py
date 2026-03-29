"""Summit-Sim Chainlit web interface.

Entry point for the Chainlit application.
"""

import asyncio
import logging
from urllib.parse import parse_qs, urlparse

import chainlit as cl  # noqa: E402
import mlflow
from chainlit import on_chat_start, on_message  # noqa: E402
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from summit_sim.graphs.utils import get_scenario_store
from summit_sim.settings import settings
from summit_sim.ui import author, simulation

logger = logging.getLogger(__name__)


class _AppState:
    """Track global app init state to avoid creating resources at import time."""

    def __init__(self) -> None:
        """Initialize state container."""
        self.initialized = False
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        """Initialize global services - only runs once per server lifespan."""
        if self.initialized:
            return

        async with self._lock:
            if self.initialized:
                return

            # 1. Initialize MLflow
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.pydantic_ai.autolog()  # type: ignore[attr-defined]
            logger.debug("MLflow initialized")

            # 2. Initialize Redis store
            await get_scenario_store()
            # 3. Initialize checkpointer indices (must be done once)
            async with AsyncRedisSaver.from_conn_string(
                settings.redis_url
            ) as checkpointer:
                await checkpointer.setup()
            logger.debug("Redis LangGraph persistence initialized")

            self.initialized = True


# Module-level state container for lazy initialization
_app_state = _AppState()


@on_chat_start
async def start() -> None:
    """Initialize chat session - routes to author or player flow."""
    await _app_state.init()  # Ensure DB indices and MLFlow are set up
    logger.info("New chat session started")
    query_string = ""
    environ = cl.context.session.environ if hasattr(cl.context, "session") else {}
    http_referer = environ.get("HTTP_REFERER", "")

    if http_referer:
        parsed = urlparse(http_referer)
        query_string = parsed.query

    params = parse_qs(query_string)
    scenario_id = params.get("scenario_id", [""])[0]

    scenario_store = await get_scenario_store()
    if scenario_id and scenario_store.get(("scenarios",), scenario_id) is not None:
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
            "Summit-Sim generates medically safe, interactive "
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


@on_message
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
