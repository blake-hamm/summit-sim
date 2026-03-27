"""Author flow handlers for the Chainlit app."""

import logging
from typing import TYPE_CHECKING

from langgraph.types import Command

from summit_sim.graphs.author import (
    ACCEPTABLE_RATING_THRESHOLD,
    MAX_RETRY_ATTEMPTS,
    AuthorState,
    create_author_graph,
)
from summit_sim.schemas import ScenarioConfig, ScenarioDraft
from summit_sim.settings import settings
from summit_sim.ui.utils import (
    get_author_form_fields,
    get_rating_actions,
    get_rating_content,
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
        content="**Configure Your Scenario**\n\nSet up your rescue simulation:",
        element=element,
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
    logger.info("Starting scenario generation")

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
            params_text = (
                f"**Focus:** {config.primary_focus} | **Env:** {config.environment} | "
                f"**Team:** {config.available_personnel} | **Evac:** "
                f"{config.evac_distance} | **Complexity:** {config.complexity}"
            )
            loading_msg.content = f"✅ *Scenario ready for review!*\n\n{params_text}"
            await loading_msg.update()
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


async def show_review_screen(state: AuthorState) -> None:
    """Display the scenario review screen with 1-5 rating buttons."""
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

    turns_sections = []
    for i, turn in enumerate(scenario.turns, 1):
        choices_lines = []
        for j, choice in enumerate(turn.choices, 1):
            correct_indicator = "✅" if choice.is_correct else "❌"
            next_info = (
                f"→ Turn {choice.next_turn_id}"
                if choice.next_turn_id is not None
                else "→ END"
            )
            choices_lines.append(
                f"   {j}. {correct_indicator} {choice.description} ({next_info})"
            )

        scene_lines = []
        if turn.scene_state:
            for key, value in turn.scene_state.items():
                scene_lines.append(f"   • {key.replace('_', ' ').title()}: {value}")
        scene_display = (
            "\n".join(scene_lines) if scene_lines else "   *No special conditions*"
        )

        hidden_lines = []
        if turn.hidden_state:
            for key, value in turn.hidden_state.items():
                hidden_lines.append(f"   • {key.replace('_', ' ').title()}: {value}")
        hidden_display = "\n".join(hidden_lines) if hidden_lines else "   *None*"

        turn_section = (
            f"**Turn {i}** (ID: {turn.turn_id})\n"
            f"{turn.narrative_text}\n\n"
            f"👁️ **Visible Scene Conditions:**\n"
            f"{scene_display}\n\n"
            f"🕵️ **Hidden State (Author View):**\n"
            f"{hidden_display}\n\n"
            f"📋 **Available Choices:**\n" + "\n".join(choices_lines)
        )
        turns_sections.append(turn_section)

    turns_content = "\n\n---\n\n".join(turns_sections)

    await cl.Message(
        content=(
            f"## {scenario.title}{attempt_text}\n"
            f"**Setting:** {scenario.setting}\n"
            f"\n**Learning Objectives:**\n"
            f"{learning_obj_text}\n"
            f"\n**Patient:** {scenario.patient_summary}\n"
            f"\n**Hidden Truth:** {scenario.hidden_truth}\n"
            f"\n**Total Turns:** {len(scenario.turns)}\n\n"
            f"---\n"
            f"### Scenario Flow\n\n"
            f"{turns_content}"
        ),
    ).send()

    res = await cl.AskActionMessage(
        content=get_rating_content(),
        actions=[
            cl.Action(name=a["name"], payload=a["payload"], label=a["label"])
            for a in get_rating_actions()
        ],
    ).send()

    if res and res.get("payload"):
        rating = res.get("payload", {}).get("value")
        if rating is not None:
            await handle_rating(state, int(rating))


async def handle_rating(state: AuthorState, rating: int) -> None:
    """Handle author rating and manage retry/approval flow."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        return

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(
            Command(resume={"rating": rating}),
            config=config_dict,
        )

        final_state = AuthorState.from_graph_result(result)
        new_retry_count = final_state.retry_count or 0

        if (
            rating < ACCEPTABLE_RATING_THRESHOLD
            and new_retry_count < MAX_RETRY_ATTEMPTS
        ):
            await cl.Message(
                content=(
                    f"🔄 Regenerating scenario "
                    f"(attempt {new_retry_count + 1}/{MAX_RETRY_ATTEMPTS})..."
                ),
            ).send()

            result = await graph.ainvoke(
                None,
                config=config_dict,
            )

            if result.get("scenario_draft"):
                new_state = AuthorState.from_graph_result(result)
                await show_review_screen(new_state)
            else:
                await cl.Message(
                    content="❌ Error: Regeneration failed. Please try again.",
                ).send()
        elif (
            rating < ACCEPTABLE_RATING_THRESHOLD
            and new_retry_count >= MAX_RETRY_ATTEMPTS
        ):
            await cl.Message(
                content=(
                    f"⚠️ Maximum retry attempts reached ({MAX_RETRY_ATTEMPTS}/"
                    f"{MAX_RETRY_ATTEMPTS}). Proceeding with current scenario."
                ),
            ).send()
            await show_completion(final_state)
        else:
            await show_completion(final_state)

    except Exception as e:
        await cl.Message(
            content=f"❌ Error during rating: {e!s}",
        ).send()
        await show_review_screen(state)


async def show_completion(state: AuthorState) -> None:
    """Display completion screen with shareable link."""
    scenario_id = state.scenario_id or ""
    shareable_url = f"{settings.base_url}?scenario_id={scenario_id}"

    await cl.Message(
        content=(
            f"#### ✅ Scenario Approved!\n\n"
            f"**Shareable URL:**\n{shareable_url}\n\n"
            f"Players can join by visiting the URL above. "
            f"The simulation is ready to run!"
        ),
    ).send()
