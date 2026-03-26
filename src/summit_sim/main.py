"""Summit-Sim Chainlit web interface.

Entry point for the Chainlit application.
"""

from urllib.parse import parse_qs, urlparse

import chainlit as cl  # noqa: E402
import mlflow
from chainlit import on_chat_start, on_message  # noqa: E402

from summit_sim.graphs.utils import scenario_store
from summit_sim.settings import settings
from summit_sim.ui import student, teacher

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)
mlflow.pydantic_ai.autolog()


@on_chat_start
async def start() -> None:
    """Initialize chat session - routes to teacher or student flow."""
    query_string = ""
    environ = cl.context.session.environ if hasattr(cl.context, "session") else {}
    http_referer = environ.get("HTTP_REFERER", "")

    if http_referer:
        parsed = urlparse(http_referer)
        query_string = parsed.query

    params = parse_qs(query_string)
    scenario_id = params.get("scenario_id", [""])[0]

    if scenario_id and scenario_store.get(("scenarios",), scenario_id) is not None:
        cl.user_session.set("mode", "student")
        cl.user_session.set("scenario_id", scenario_id)
        await student.start_student_session()
        return

    if scenario_id:
        await cl.Message(
            content=(
                "❌ Scenario not found. "
                "Please check the URL or ask your teacher for a valid link."
            ),
        ).send()
        return

    cl.user_session.set("mode", "teacher")
    await teacher.ask_scenario_config()


@on_message
async def on_message_handler(_message: cl.Message) -> None:
    """Handle incoming messages (fallback handler)."""
    mode = cl.user_session.get("mode")

    if mode == "teacher":
        await cl.Message(
            content=(
                "Type **restart** to create a new scenario, or use the buttons above."
            ),
        ).send()
    elif mode == "student":
        await cl.Message(
            content="Please use the buttons above to make your selection.",
        ).send()
    else:
        await cl.Message(
            content="Welcome! The scenario creator will start automatically.",
        ).send()
        await start()
