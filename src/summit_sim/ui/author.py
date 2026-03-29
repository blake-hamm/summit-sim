"""Author flow handlers for the Chainlit app."""

import logging
from typing import TYPE_CHECKING

import mlflow
from langgraph.types import Command
from mlflow.entities import AssessmentSource, AssessmentSourceType

from summit_sim.graphs.author import (
    MAX_RETRY_ATTEMPTS,
    AuthorState,
    create_author_graph,
)
from summit_sim.graphs.utils import get_scenario_store
from summit_sim.schemas import ScenarioConfig, ScenarioDraft
from summit_sim.settings import settings
from summit_sim.ui import simulation
from summit_sim.ui.utils import (
    format_scenario_intro,
    get_author_form_fields,
    get_review_actions,
    get_review_content,
    get_satisfaction_actions,
    get_satisfaction_content,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import chainlit as cl
    from langchain_core.runnables import RunnableConfig
else:
    import chainlit as cl


async def ask_scenario_config() -> None:
    """Ask for scenario configuration using a form."""
    element = cl.CustomElement(
        name="ScenarioConfigForm",
        display="inline",
        props={"fields": get_author_form_fields()},
    )

    res = await cl.AskElementMessage(
        content="",
        element=element,
        timeout=settings.ui_timeout,
    ).send()

    if res and res.get("submitted"):
        primary_focus = res.get("primary_focus")
        environment = res.get("environment")
        available_personnel = res.get("available_personnel")
        evac_distance = res.get("evac_distance")
        complexity = res.get("complexity")

        if not all(
            [primary_focus, environment, available_personnel, evac_distance, complexity]
        ):
            raise ValueError("Missing required scenario configuration values")

        cl.user_session.set("primary_focus", primary_focus)
        cl.user_session.set("environment", environment)
        cl.user_session.set("available_personnel", available_personnel)
        cl.user_session.set("evac_distance", evac_distance)
        cl.user_session.set("complexity", complexity)
        await generate_scenario()


async def generate_scenario() -> None:
    """Generate scenario with collected config."""
    primary_focus = cl.user_session.get("primary_focus")
    environment = cl.user_session.get("environment")
    available_personnel = cl.user_session.get("available_personnel")
    evac_distance = cl.user_session.get("evac_distance")
    complexity = cl.user_session.get("complexity")

    if not all(
        [primary_focus, environment, available_personnel, evac_distance, complexity]
    ):
        raise ValueError(
            "Missing required scenario config in session. "
            "Did you navigate to this page without completing the form?"
        )

    config = ScenarioConfig.model_validate(
        {
            "primary_focus": primary_focus,
            "environment": environment,
            "available_personnel": available_personnel,
            "evac_distance": evac_distance,
            "complexity": complexity,
        }
    )

    cl.user_session.set("scenario_config", config)

    loading_msg = await cl.Message(content="⏳ *Generating your scenario...*").send()

    graph = create_author_graph()
    cl.user_session.set("graph", graph)

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    initial_state: AuthorState = AuthorState(
        scenario_config=config.model_dump(),
        scenario_draft=None,
        scenario_id="",
        retry_count=0,
        approval_status=None,
    )

    try:
        result = await graph.ainvoke(
            initial_state,
            config=config_dict,
        )

        if result.get("scenario_draft"):
            state = AuthorState.from_graph_result(result)
            mode = cl.user_session.get("mode", "instructor")
            is_student = mode == "student"
            params_text = (
                f"**Focus:** {config.primary_focus}\n**Env:** {config.environment}\n"
                f"**Team:** {config.available_personnel}\n**Evac:** "
                f"{config.evac_distance}\n**Complexity:** {config.complexity}"
            )
            loading_msg.content = f"✅ *Scenario ready for review!*\n\n{params_text}"
            await loading_msg.update()

            # Route students directly to simulation, instructors to review screen
            if is_student:
                await handle_student_start(state)
            else:
                await show_review_screen(state)
        else:
            loading_msg.content = (
                "❌ Error: Scenario generation failed. Please try again."
            )
            await loading_msg.update()
            return

    except Exception as e:
        loading_msg.content = f"❌ Error during generation: {e!s}"
        await loading_msg.update()


async def show_completion(state: AuthorState) -> None:
    """Display completion screen with shareable link and collect rating."""
    scenario_id = state.scenario_id or ""
    shareable_url = f"{settings.base_url}?scenario_id={scenario_id}"

    await cl.Message(
        content=(
            f"Players can join by visiting the link below. "
            f"The simulation is ready to run!\n\n"
            f"**🔗 Shareable URL:** {shareable_url}"
        ),
    ).send()

    # Ask for satisfaction rating (non-blocking)
    res = await cl.AskActionMessage(
        content=get_satisfaction_content(),
        actions=[
            cl.Action(name=a["name"], payload=a["payload"], label=a["label"])
            for a in get_satisfaction_actions()
        ],
        timeout=settings.ui_timeout,
    ).send()

    if res and res.get("payload"):
        rating = res.get("payload", {}).get("value")
        if rating is not None and state.current_trace_id:
            # Log rating to MLflow using the trace_id from the graph state
            mlflow.log_feedback(
                trace_id=state.current_trace_id,
                name="author_rating",
                value=rating,
                rationale="Author satisfaction rating for the scenario",
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=state.scenario_id,
                ),
            )


async def handle_student_start(_state: AuthorState) -> None:
    """Handle student mode - auto-approve and start simulation immediately."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        return

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        # Auto-approve the scenario
        result = await graph.ainvoke(
            Command(resume={"action": "approve"}),
            config=config_dict,
        )

        final_state = AuthorState.from_graph_result(result)
        scenario_id = final_state.scenario_id or ""

        # Set up simulation session
        cl.user_session.set("scenario_id", scenario_id)
        cl.user_session.set("mode", "player")
        # Store trace_id from authoring phase for correlation with simulation traces
        if final_state.current_trace_id:
            cl.user_session.set("authoring_trace_id", final_state.current_trace_id)

        # Load scenario from store (it was just saved during approval)
        store = await get_scenario_store()
        store_result = await store.aget(("scenarios",), scenario_id)
        if store_result is None:
            await cl.Message(
                content="❌ Error: Scenario not found in store.",
            ).send()
            return

        scenario_data = store_result.value
        scenario = ScenarioDraft.model_validate(scenario_data["scenario_draft"])
        cl.user_session.set("scenario", scenario)

        # Show scenario context for student before starting
        context_content = format_scenario_intro(scenario)

        await cl.Message(
            content=context_content,
            elements=[
                cl.CustomElement(
                    name="ChatEnabler",
                    props={},
                    display="inline",
                )
            ],
        ).send()

        # Start simulation (skip intro since we just showed context)
        await simulation.run_simulation()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error starting simulation: {e!s}",
        ).send()


async def handle_approval(_state: AuthorState) -> None:
    """Handle scenario approval and save to store."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        return

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(
            Command(resume={"action": "approve"}),
            config=config_dict,
        )

        final_state = AuthorState.from_graph_result(result)
        await show_completion(final_state)

    except Exception as e:
        await cl.Message(
            content=f"❌ Error during approval: {e!s}",
        ).send()


async def handle_revision(state: AuthorState) -> None:
    """Handle revision request with human-in-the-loop feedback."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        return

    # Create feedback form element
    element = cl.CustomElement(name="RevisionFeedbackForm", props={})

    # Ask author for revision feedback using form
    res = await cl.AskElementMessage(
        content="Please provide specific feedback on what you'd like changed:",
        element=element,
        timeout=settings.ui_timeout,
    ).send()

    if not res or not res.get("submitted"):
        await cl.Message(
            content="❌ No feedback provided. Returning to review screen.",
        ).send()
        await show_review_screen(state)
        return

    feedback = res.get("output")

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        # Resume graph with revision feedback
        result = await graph.ainvoke(
            Command(resume={"action": "revise", "feedback": feedback}),
            config=config_dict,
        )

        final_state = AuthorState.from_graph_result(result)
        new_retry_count = final_state.retry_count or 0

        if new_retry_count >= MAX_RETRY_ATTEMPTS:
            await cl.Message(
                content=(
                    f"⚠️ Maximum revision attempts reached ({MAX_RETRY_ATTEMPTS}/"
                    f"{MAX_RETRY_ATTEMPTS}). Proceeding with current scenario."
                ),
            ).send()
            await show_completion(final_state)
        else:
            await cl.Message(
                content=(
                    f"🔄 Revising scenario based on your feedback "
                    f"(attempt {new_retry_count}/{MAX_RETRY_ATTEMPTS})..."
                ),
            ).send()

            # Continue graph to regenerate
            result = await graph.ainvoke(
                None,
                config=config_dict,
            )

            if result.get("scenario_draft"):
                new_state = AuthorState.from_graph_result(result)
                await show_review_screen(new_state)
            else:
                await cl.Message(
                    content="❌ Error: Revision failed. Please try again.",
                ).send()

    except Exception as e:
        await cl.Message(
            content=f"❌ Error during revision: {e!s}",
        ).send()
        await show_review_screen(state)


async def show_review_screen(state: AuthorState) -> None:
    """Display the scenario review screen with approve/revise buttons."""
    scenario_dict = state.scenario_draft
    retry_count = state.retry_count

    if scenario_dict is None:
        await cl.Message(
            content="❌ Error: No scenario to review.",
        ).send()
        return

    scenario = ScenarioDraft.model_validate(scenario_dict)

    learning_obj_text = "\n".join(f"• {obj}" for obj in scenario.learning_objectives)
    attempt_text = f" (Attempt {retry_count + 1}/3)" if retry_count > 0 else ""

    scene_display = (
        scenario.scene_state if scenario.scene_state else "*No special conditions*"
    )

    content = f"""## 🏔️ {scenario.title}{attempt_text}

#### 🎯 Learning Objectives
{learning_obj_text}

#### 🏔️ Environment
**Setting:** {scenario.setting}

**Scene State:** {scene_display}

#### 🏥 Patient
**Summary:** {scenario.patient_summary}

**Opening Narrative:** {scenario.initial_narrative}

#### 🔒 Instructor Only
**Hidden Truth:** {scenario.hidden_truth}

**Hidden State:** {scenario.hidden_state}"""

    await cl.Message(content=content).send()

    res = await cl.AskActionMessage(
        content=get_review_content(),
        actions=[
            cl.Action(name=a["name"], payload=a["payload"], label=a["label"])
            for a in get_review_actions()
        ],
        timeout=settings.ui_timeout,
    ).send()

    if res and res.get("payload"):
        action = res.get("payload", {}).get("action")
        if action == "approve":
            await handle_approval(state)
        elif action == "revise":
            await handle_revision(state)
