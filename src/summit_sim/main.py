"""Summit-Sim Chainlit web interface.

Entry point for the Chainlit application.
"""

import logging
from urllib.parse import parse_qs, urlparse

import chainlit as cl  # noqa: E402
import mlflow
from chainlit import on_chat_start, on_message  # noqa: E402

from summit_sim.graphs.utils import scenario_store
from summit_sim.settings import settings
from summit_sim.ui import author, simulation

logger = logging.getLogger(__name__)


class _MLflowState:
    """Track MLflow initialization state to avoid creating resources at import time."""

    def __init__(self) -> None:
        """Initialize state container."""
        self.initialized = False

    def init(self) -> None:
        """Initialize MLflow - only runs once when chat session starts."""
        if not self.initialized:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.pydantic_ai.autolog()  # type: ignore[attr-defined]
            self.initialized = True
            logger.debug("MLflow initialized")


# Module-level state container for lazy initialization
_mlflow_state = _MLflowState()


@on_chat_start
async def start() -> None:
    """Initialize chat session - routes to author or player flow."""
    _mlflow_state.init()
    logger.info("New chat session started")
    query_string = ""
    environ = cl.context.session.environ if hasattr(cl.context, "session") else {}
    http_referer = environ.get("HTTP_REFERER", "")

    if http_referer:
        parsed = urlparse(http_referer)
        query_string = parsed.query

    params = parse_qs(query_string)
    scenario_id = params.get("scenario_id", [""])[0]

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

    logger.info("Starting author flow")
    cl.user_session.set("mode", "author")
    await author.ask_scenario_config()


@on_message
async def on_message_handler(_message: cl.Message) -> None:
    """Handle incoming messages (fallback handler)."""
    mode = cl.user_session.get("mode")

    if mode == "author":
        await cl.Message(
            content=(
                "Type **restart** to create a new scenario, or use the buttons above."
            ),
        ).send()
    elif mode == "player":
        await cl.Message(
            content="Please use the buttons above to make your selection.",
        ).send()
    else:
        await cl.Message(
            content="Welcome! The scenario creator will start automatically.",
        ).send()
        await start()
