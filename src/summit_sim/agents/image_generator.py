"""Async image generation for scenario visuals using OpenRouter."""

import logging

import httpx
import mlflow
from mlflow.entities import SpanType

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
    """Build image generation prompt from scenario fields and optional config.

    Uses the scenario title and setting directly without fragile parsing.
    The setting field already contains rich environmental description including
    terrain, weather, and time of day cues. If config is provided, includes
    scenario context like environment type, group size, and complexity.
    """
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

    Args:
        scenario: The scenario to generate an image for
        config: Optional ScenarioConfig with context (environment, group size, etc.)
        model: OpenRouter model name (defaults to settings.image_generation_model)

    Returns:
        Base64-encoded image string or None if generation fails

    """
    model = model or settings.image_generation_model
    prompt = build_image_prompt(scenario, config)

    logger.info(
        "Generating scenario image: scenario_id=%s, model=%s",
        scenario.title,
        model,
    )

    # Set span attributes for cost tracking
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_attributes(
            {
                "image.model": model,
                "image.prompt_length": len(prompt),
            }
        )

    try:
        async with httpx.AsyncClient(
            timeout=settings.image_generation_timeout
        ) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "modalities": ["image"],
                    "image_config": {
                        "aspect_ratio": "16:9",  # 1344×768 landscape, mobile-optimized
                        "image_size": "1K",
                    },
                    "reasoning": {"effort": "none"},  # Disables reasoning entirely
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract image from response
            # Format: choices[0].message.images[].image_url.url (base64 data URL)
            choices = data.get("choices", [])
            if not choices:
                logger.warning("No choices in image generation response")
                return None

            message = choices[0].get("message", {})
            images = message.get("images", [])

            if not images:
                logger.warning("No images in response message")
                return None

            image_url = images[0].get("image_url", {}).get("url", "")

            # Handle base64 data URL
            if image_url.startswith("data:image"):
                # Strip prefix and return base64 string
                base64_data = image_url.split(",")[1]
                logger.info("Successfully generated image: %d bytes", len(base64_data))

                # Set span attributes for successful generation
                if active_span:
                    active_span.set_attributes(
                        {
                            "image.size_bytes": len(base64_data),
                            "image.success": True,
                        }
                    )

                return base64_data

            logger.warning("Unexpected image URL format")

            # Set span attributes for unexpected format
            if active_span:
                active_span.set_attributes(
                    {
                        "image.success": False,
                        "image.error": "unexpected_format",
                    }
                )

            return None

    except httpx.TimeoutException:
        logger.warning(
            "Image generation timed out after %ds", settings.image_generation_timeout
        )

        # Set span attributes for timeout
        if active_span:
            active_span.set_attributes(
                {
                    "image.success": False,
                    "image.error": "timeout",
                }
            )

        return None

    except Exception as e:
        logger.warning("Image generation failed: %s", e)

        # Set span attributes for general failure
        if active_span:
            active_span.set_attributes(
                {
                    "image.success": False,
                    "image.error": str(e),
                }
            )

        return None
