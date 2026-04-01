"""Async image generation for scenario visuals using OpenRouter."""

import logging

import httpx
import mlflow
from mlflow.entities import SpanType

from summit_sim.schemas import ScenarioConfig, ScenarioDraft
from summit_sim.settings import settings

logger = logging.getLogger(__name__)

IMAGE_PROMPT_TEMPLATE = """\
Create a realistic cinematic wilderness scene for a wilderness first responder \
training scenario titled "{title}".

Setting: {setting}

Scenario Context:
- Environment Type: {environment}
- Group Size: {available_personnel}
- Evacuation Distance: {evac_distance}
- Complexity Level: {complexity}
- Primary Focus: {primary_focus}

The image should feel immersive, grounded, and educational, with natural \
lighting, clear environment cues, and a strong sense of place. Avoid text, \
labels, logos, gore, exaggerated fantasy elements, or obvious AI-art artifacts. \
Do not depict hidden medical details or anything that reveals the correct diagnosis.
Do not include any text in the image.

Optimize for mobile viewing: strong focal point, simple composition, clear \
readability on small screens."""


def build_image_prompt(
    scenario: ScenarioDraft,
    config: ScenarioConfig | None = None,
) -> str:
    """Build image generation prompt from scenario fields and optional config.

    Uses the scenario title and setting directly without fragile parsing.
    The setting field already contains rich environmental description including
    terrain, weather, and time of day cues. If config is provided, includes
    scenario context like environment type, group size, and complexity.
    """
    if config:
        return IMAGE_PROMPT_TEMPLATE.format(
            title=scenario.title,
            setting=scenario.setting,
            environment=config.environment,
            available_personnel=config.available_personnel,
            evac_distance=config.evac_distance,
            complexity=config.complexity,
            primary_focus=config.primary_focus,
        )
    # Fallback to basic prompt without config context
    return IMAGE_PROMPT_TEMPLATE.format(
        title=scenario.title,
        setting=scenario.setting,
        environment="Unknown",
        available_personnel="Unknown",
        evac_distance="Unknown",
        complexity="Unknown",
        primary_focus="Unknown",
    )


@mlflow.trace(span_type=SpanType.LLM)
async def generate_scenario_image(
    scenario: ScenarioDraft,
    config: ScenarioConfig | None = None,
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

    # Log metadata for debugging and cost tracking
    # mlflow.log_param("image_model", model)
    # mlflow.log_param("prompt_length", len(prompt))

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
                        "aspect_ratio": "16:9"  # 1344×768 landscape, mobile-optimized
                    },
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
                return base64_data
            logger.warning("Unexpected image URL format")
            return None

    except httpx.TimeoutException:
        logger.warning(
            "Image generation timed out after %ds", settings.image_generation_timeout
        )
        return None
    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return None
