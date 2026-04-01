# Phased Implementation Plan: Async Scenario Image Generation

## Overview

Generate atmospheric, mobile-friendly wilderness scene images for Summit-Sim scenarios using OpenRouter's image-capable models. Images enhance immersion on first load, especially on mobile where no image currently appears.

**Model**: `bytedance/seedream-4.5` (configurable via settings)  
**Dimensions**: 1024x768 (landscape, mobile-optimized)  
**Storage**: LangGraph store (base64 in ScenarioDraft)  
**UI**: Bytes passed to `cl.Image(content=...)`  

---

## Phase 1: MVP - Core Image Generation (Notebook Testing)

**Goal**: Minimum code to test image generation in a notebook. Skip graph integration and UI for now.

### Files to Create/Modify

#### 1. New: `src/summit_sim/agents/image_generator.py`

Core async image generation logic:

```python
"""Async image generation for scenario visuals using OpenRouter."""

import base64
import logging
from typing import Any

import httpx
import mlflow
from mlflow.entities import SpanType

from summit_sim.schemas import ScenarioDraft
from summit_sim.settings import settings

logger = logging.getLogger(__name__)

IMAGE_PROMPT_TEMPLATE = """\
Create a realistic cinematic wilderness scene for a wilderness first responder training scenario. 
Show {setting}, with {terrain}, {weather}, and {time_of_day}. 
Mood: {overall_mood}. 

The image should feel immersive, grounded, and educational, with natural lighting, clear environment cues, and a strong sense of place. Avoid text, labels, logos, gore, exaggerated fantasy elements, or obvious AI-art artifacts. Do not depict hidden medical details or anything that reveals the correct diagnosis.

Optimize for mobile viewing: strong focal point, simple composition, clear readability on small screens."""


def _extract_scene_details(scene_state: str) -> dict[str, str]:
    """Parse scene_state for terrain, weather, time cues."""
    scene_lower = scene_state.lower()
    
    # Terrain detection
    terrain = "natural wilderness terrain"
    if any(word in scene_lower for word in ["mountain", "peak", "ridge", "alpine"]):
        terrain = "mountainous alpine terrain with rocky outcrops"
    elif any(word in scene_lower for word in ["forest", "wood", "tree"]):
        terrain = "dense forest with natural clearings"
    elif any(word in scene_lower for word in ["desert", "sand", "arid"]):
        terrain = "arid desert landscape with sparse vegetation"
    elif any(word in scene_lower for word in ["snow", "winter", "ice"]):
        terrain = "snow-covered winter wilderness"
    elif any(word in scene_lower for word in ["river", "water", "lake"]):
        terrain = "riverside or lakeside wilderness setting"
    
    # Weather detection
    weather = "clear conditions"
    if any(word in scene_lower for word in ["storm", "rain", "thunder"]):
        weather = "stormy weather with dark clouds"
    elif any(word in scene_lower for word in ["snow", "blizzard", "whiteout"]):
        weather = "snowy conditions with reduced visibility"
    elif any(word in scene_lower for word in ["wind", "gust"]):
        weather = "windy conditions"
    elif any(word in scene_lower for word in ["fog", "mist", "cloud"]):
        weather = "foggy or overcast conditions"
    elif any(word in scene_lower for word in ["hot", "heat", "sun"]):
        weather = "sunny and warm conditions"
    
    # Time of day detection
    time_of_day = "daylight hours"
    if any(word in scene_lower for word in ["dusk", "sunset", "evening"]):
        time_of_day = "dusk or sunset"
    elif any(word in scene_lower for word in ["dawn", "sunrise", "morning"]):
        time_of_day = "dawn or sunrise"
    elif any(word in scene_lower for word in ["night", "dark", "midnight"]):
        time_of_day = "nighttime with limited visibility"
    
    return {
        "terrain": terrain,
        "weather": weather,
        "time_of_day": time_of_day,
    }


def _infer_mood(scenario: ScenarioDraft) -> str:
    """Infer overall mood from scenario title and content."""
    title_lower = scenario.title.lower()
    setting_lower = scenario.setting.lower()
    
    # Critical/urgent scenarios
    if any(word in title_lower for word in ["critical", "emergency", "rescue", "stranded"]):
        return "tense and urgent"
    
    # Environmental hazards
    if any(word in title_lower for word in ["storm", "lightning", "avalanche", "flood"]):
        return "dramatic and challenging"
    
    # Medical emergencies
    if any(word in title_lower for word in ["anaphylaxis", "attack", "bleeding", "injury"]):
        return "serious and focused"
    
    # Altitude/environmental
    if any(word in title_lower for word in ["altitude", "hypothermia", "heat", "cold"]):
        return "harsh and demanding"
    
    # Default
    return "authentic wilderness atmosphere"


def build_image_prompt(scenario: ScenarioDraft) -> str:
    """Build image generation prompt from scenario fields."""
    scene_details = _extract_scene_details(scenario.scene_state)
    overall_mood = _infer_mood(scenario)
    
    return IMAGE_PROMPT_TEMPLATE.format(
        setting=scenario.setting,
        terrain=scene_details["terrain"],
        weather=scene_details["weather"],
        time_of_day=scene_details["time_of_day"],
        overall_mood=overall_mood,
    )


@mlflow.trace(span_type=SpanType.LLM)
async def generate_scenario_image(
    scenario: ScenarioDraft,
    model: str | None = None,
) -> bytes | None:
    """Generate atmospheric wilderness scene image for scenario.
    
    Returns raw image bytes or None on failure. Non-blocking - exceptions 
    are caught and logged. Image is 1024x768 landscape optimized for mobile.
    
    Args:
        scenario: The scenario to generate an image for
        model: OpenRouter model name (defaults to settings.image_generation_model)
        
    Returns:
        Raw image bytes or None if generation fails
    """
    model = model or settings.image_generation_model
    prompt = build_image_prompt(scenario)
    
    logger.info(
        "Generating scenario image: scenario_id=%s, model=%s",
        scenario.title,
        model,
    )
    
    try:
        async with httpx.AsyncClient(timeout=settings.image_generation_timeout) as client:
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
                    "modalities": ["image", "text"],
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
                # Strip prefix and decode
                base64_data = image_url.split(",")[1]
                image_bytes = base64.b64decode(base64_data)
                logger.info("Successfully generated image: %d bytes", len(image_bytes))
                return image_bytes
            else:
                logger.warning("Unexpected image URL format")
                return None
                
    except httpx.TimeoutException:
        logger.warning("Image generation timed out after %ds", settings.image_generation_timeout)
        return None
    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return None
```

#### 2. Modify: `src/summit_sim/settings.py`

Add image generation configuration:

```python
image_generation_model: str = Field(
    default="bytedance/seedream-4.5",
    description="OpenRouter model for scenario image generation",
)
image_generation_timeout: int = Field(
    default=60,  # Image generation can be slow
    description="Timeout in seconds for image generation",
)
```

#### 3. New: `notebooks/test_image_generation.ipynb`

Create a testing notebook that:
1. Generates a scenario using existing code
2. Tests `build_image_prompt()` function
3. Calls `generate_scenario_image()` 
4. Displays the image inline using IPython.display
5. Tests error handling (simulate failures)
6. Tests multiple scenario types (mountain, forest, desert, etc.)

**Template structure**:
```python
# Cell 1: Setup
import warnings
warnings.filterwarnings("ignore")

from summit_sim.agents.generator import generate_scenario
from summit_sim.agents.image_generator import build_image_prompt, generate_scenario_image
from summit_sim.schemas import ScenarioConfig

# Cell 2: Generate a scenario
config = ScenarioConfig(
    primary_focus="Trauma",
    environment="Alpine/Mountain",
    available_personnel="Small Group (3-5)",
    evac_distance="Remote (1 day)",
    complexity="Standard",
)

scenario = await generate_scenario(config)
print(f"Generated: {scenario.title}")

# Cell 3: Build and display prompt
prompt = build_image_prompt(scenario)
print(f"\nImage Prompt:\n{prompt}")

# Cell 4: Generate image
image_bytes = await generate_scenario_image(scenario)
if image_bytes:
    print(f"✓ Generated image: {len(image_bytes)} bytes")
    # Display using IPython
    from IPython.display import Image, display
    display(Image(data=image_bytes))
else:
    print("✗ Image generation failed (non-blocking)")

# Cell 5: Test multiple scenarios
# Generate 3-4 different scenarios and compare images
```

### Acceptance Criteria for Phase 1

- [ ] `image_generator.py` module created with async generation logic
- [ ] Settings updated with image model config
- [ ] Notebook can generate scenarios and display images
- [ ] Image generation is truly non-blocking (returns None on failure)
- [ ] Prompts are well-formed and scenario-appropriate
- [ ] All code passes ruff linting and formatting

---

## Phase 2: Integration (After Notebook Validation)

**Goal**: Integrate image generation into the author graph and UI. Only proceed after Phase 1 notebook testing is successful.

### Files to Modify

#### 1. `src/summit_sim/schemas.py`

Add `image_data` field to ScenarioDraft:

```python
image_data: bytes | None = Field(
    default=None,
    description="Raw image bytes for scenario visualization",
)
```

#### 2. `src/summit_sim/graphs/author.py`

Add image generation node to graph:

```python
async def generate_image_node(state: AuthorState, config: RunnableConfig) -> dict:
    """Generate scenario image after saving to store.
    
    Non-blocking - if generation fails, scenario is still usable.
    Regenerates image on revision (new scenario draft).
    """
    if not state.scenario_draft:
        return {}
    
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    
    # Generate image
    image_bytes = await generate_scenario_image(scenario)
    
    if image_bytes:
        # Update scenario with image
        scenario.image_data = image_bytes
        
        # Update stored scenario
        store = AppState.store
        if store:
            await store.aput(
                ("scenarios",),
                state.scenario_id,
                {"scenario_draft": scenario.model_dump(mode="json")},
            )
        
        logger.info(
            "Image generated and saved: scenario_id=%s, size=%d bytes",
            state.scenario_id,
            len(image_bytes),
        )
    else:
        logger.warning(
            "Image generation failed (non-blocking): scenario_id=%s",
            state.scenario_id,
        )
    
    return {"scenario_draft": scenario.model_dump()}


# In create_author_graph():
workflow.add_node("generate_image", generate_image_node)
workflow.add_edge("save", "generate_image")  # After save, generate image
workflow.add_edge("generate_image", END)
```

**Note**: Image regenerates on revision because `generate_image_node` runs after `save`, and `save` runs after each successful generation (including revisions).

#### 3. `src/summit_sim/ui/simulation.py`

Update `show_scenario_intro()` to display images:

```python
async def show_scenario_intro(scenario: ScenarioDraft) -> None:
    """Display scenario intro with image or placeholder."""
    content = format_scenario_intro(scenario)
    
    elements = []
    
    if scenario.image_data:
        # Display actual image
        elements.append(cl.Image(
            content=scenario.image_data,  # Raw bytes
            name="scenario_image",
            display="inline",
            size="large",
        ))
    else:
        # Show placeholder
        elements.append(cl.Text(
            content="🖼️ Generating scenario image...",
            display="inline",
        ))
    
    await cl.Message(content=content, elements=elements).send()
    
    await run_simulation()
```

#### 4. `src/summit_sim/ui/author.py`

Update `handle_student_start()` and `handle_approval()` to pass scenario with image to simulation:

The scenario is already loaded from store in these functions, so image_data will automatically be available if it was generated.

### Testing Strategy for Phase 2

1. **End-to-End Test**: Create scenario → approve → verify image appears in student view
2. **Revision Test**: Revise scenario → verify new image is generated
3. **Failure Test**: Temporarily break image generation → verify scenario still loads
4. **Mobile Test**: Verify image displays correctly on mobile (1024x768 ratio)

### Acceptance Criteria for Phase 2

- [ ] Scenario image stored in LangGraph store with scenario
- [ ] Image displays inline in Chainlit UI
- [ ] Placeholder shown while image generating
- [ ] Image regenerates on scenario revision
- [ ] Scenario loads successfully even if image generation fails
- [ ] All tests pass (coverage >= 80%)
- [ ] Code passes quality gates

---

## Implementation Notes

### Design Decisions

1. **Async httpx**: Using `httpx.AsyncClient` instead of PydanticAI because image generation via chat completions with modalities is a different API pattern than structured outputs.

2. **Bytes vs Base64**: Store raw bytes in ScenarioDraft (Chainlit accepts bytes via `content` parameter). Base64 decoding happens only once in `generate_scenario_image()`.

3. **Non-blocking Strategy**: Image generation runs after `save_scenario` so the scenario is already persisted. If image fails, the scenario is still usable. The UI shows a placeholder initially, then the image appears on reload if generation completes.

4. **Regeneration on Revision**: The graph flow `save → generate_image` means every time we save (including after revisions), we regenerate the image. This ensures the image matches the revised scenario.

5. **Cost Tracking**: Image generation costs are already tracked via MLflow tracing (enabled via `@mlflow.trace` decorator).

### Mobile Optimization

The prompt template includes specific instructions:
- Strong focal point
- Simple composition
- Clear readability on small screens
- 1024x768 landscape aspect ratio

This ensures images look good on mobile devices where they're most needed.

### Error Handling Philosophy

Following Summit-Sim conventions:
- Fail fast, don't hide errors
- Log warnings for recoverable failures (image generation)
- Never block core functionality (scenario loading) on optional features
- Let exceptions propagate for truly unexpected errors

---

## Future Enhancements (Post-MVP)

1. **Image Caching**: Cache generated images to avoid regenerating identical scenarios
2. **Batch Generation**: Generate multiple images and pick best one
3. **Style Consistency**: Fine-tune prompts for consistent visual style across scenarios
4. **Progressive Loading**: Show low-res thumbnail first, then full image
5. **Accessibility**: Add alt text describing the scene for screen readers

---

## Related Files

- `src/summit_sim/agents/generator.py` - Scenario generation (Phase 1 dependency)
- `src/summit_sim/schemas.py` - ScenarioDraft schema
- `src/summit_sim/settings.py` - Configuration
- `src/summit_sim/graphs/author.py` - Author workflow graph
- `src/summit_sim/ui/simulation.py` - Student simulation UI
- `src/summit_sim/ui/author.py` - Author flow handlers

---

**Status**: Ready for Phase 1 implementation  
**Priority**: Medium  
**Dependencies**: None (Phase 1), Phase 1 success (Phase 2)
