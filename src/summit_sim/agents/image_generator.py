"""Async image generation for scenario visuals using Vertex AI."""

import base64
import logging

import mlflow
from google.genai import types
from mlflow.entities import SpanType

from summit_sim.agents.utils import get_provider
from summit_sim.schemas import ScenarioConfig, ScenarioDraft
from summit_sim.settings import settings

logger = logging.getLogger(__name__)

IMAGE_PROMPT_TEMPLATE = """\
Create a high-quality, cinematic digital painting of a wilderness first responder \
rescue in progress, representing a scenario titled "{title}". Use a serious, \
atmospheric concept art style (similar to a survival video game).

Setting: {setting}

Scenario Context:
- Environment Type: {environment}
- Group Size: {available_personnel} (Ensure distinct figures)
- Evacuation Distance: {evac_distance}
- Complexity Level: {complexity}
- Primary Focus: {primary_focus}

Action: Show one patient on the ground being assessed by the rescuers. \
The image should feel immersive, grounded, and educational, featuring atmospheric \
lighting, painterly brushstrokes, clear environment cues, and a strong sense of place.

Avoid photorealism, text, labels, logos, gore, exaggerated fantasy elements, \
or messy AI-art artifacts like fused bodies or extra limbs. \
Do not depict hidden medical details, open wounds, or anything that reveals \
the correct diagnosis. Do not include any text in the image.

Ensure strong central focal point, simple uncluttered \
composition, wide camera angle, and clear readability on small screens.
"""


def build_image_prompt(
    scenario: ScenarioDraft,
    config: ScenarioConfig,
) -> str:
    """Build image generation prompt from scenario fields and optional config."""
    return IMAGE_PROMPT_TEMPLATE.format(
        title=scenario.title,
        setting=scenario.setting,
        environment=config.environment,
        available_personnel=config.available_personnel,
        evac_distance=config.evac_distance,
        complexity=config.complexity,
        primary_focus=config.primary_focus,
    )


@mlflow.trace(span_type=SpanType.AGENT)
async def generate_scenario_image(
    scenario: ScenarioDraft,
    config: ScenarioConfig,
    model: str | None = None,
) -> str | None:
    """Generate atmospheric wilderness scene image for scenario.

    Returns base64-encoded image string or None on failure. Non-blocking - exceptions
    are caught and logged. Image is 1344×768 (16:9 aspect ratio) landscape
    optimized for mobile.
    """
    model = model or settings.image_generation_model
    prompt = build_image_prompt(scenario, config)

    logger.info(
        "Generating scenario image: scenario_id=%s, model=%s",
        scenario.title,
        model,
    )

    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_attributes(
            {
                "image.model": model,
                "image.prompt_length": len(prompt),
            }
        )

    provider = get_provider()

    try:
        response = await provider.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="16:9"),
            ),
        )

        if not response.candidates or not response.candidates[0].content:
            logger.warning("No candidates in response")
            if active_span:
                active_span.set_attributes(
                    {
                        "image.success": False,
                        "image.error": "no_candidates",
                    }
                )
            return None

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                image_data = part.inline_data.data
                if isinstance(image_data, bytes):
                    logger.info(
                        "Successfully generated image: %d bytes", len(image_data)
                    )
                    if active_span:
                        active_span.set_attributes(
                            {
                                "image.size_bytes": len(image_data),
                                "image.success": True,
                            }
                        )
                    return base64.b64encode(image_data).decode("ascii")

        logger.warning("No image in response")
        if active_span:
            active_span.set_attributes(
                {
                    "image.success": False,
                    "image.error": "no_image",
                }
            )
        return None

    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        if active_span:
            active_span.set_attributes(
                {
                    "image.success": False,
                    "image.error": str(e),
                }
            )
        return None
