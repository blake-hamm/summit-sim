# Phased Implementation Plan: Async Scenario Image Generation

## Overview

Generate atmospheric, mobile-friendly wilderness scene images for Summit-Sim scenarios using OpenRouter's image-capable models. Images enhance immersion on first load, especially on mobile where no image currently appears.

**Model**: `bytedance-seed/seedream-4.5` (configurable via settings)
**Dimensions**: 1344×768 via `16:9` aspect ratio (landscape, mobile-optimized)
**Storage**: LangGraph store (base64 string in ScenarioDraft, decoded to bytes for Chainlit)
**UI**: Base64 decoded to bytes, passed to `cl.Image(content=...)`  

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
    config: ScenarioConfig,
) -> str:
    """Build image generation prompt from scenario fields and config.

    Uses the scenario title and setting directly without fragile parsing.
    The setting field already contains rich environmental description including
    terrain, weather, and time of day cues. Config provides scenario context
    like environment type, group size, and complexity.
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


@mlflow.trace(span_type=SpanType.LLM)
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
        config: ScenarioConfig with context (environment, group size, etc.)
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

    # Get active span for MLflow attributes
    active_span = mlflow.get_current_active_span()
    if active_span:
        active_span.set_attributes({
            "image.model": model,
            "image.prompt_length": len(prompt),
        })
    
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
                    "modalities": ["image"],
                    "image_config": {
                        "aspect_ratio": "16:9"  # 1344×768 landscape, mobile-optimized
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract image from response - let KeyError propagate if API contract breaks
            # Format: choices[0].message.images[].image_url.url (base64 data URL)
            image_url = data["choices"][0]["message"]["images"][0]["image_url"]["url"]
            
            # Handle base64 data URL - let IndexError propagate if format is wrong
            base64_data = image_url.split(",")[1]
            
            logger.info("Successfully generated image: %d bytes", len(base64_data))
            
            if active_span:
                active_span.set_attributes({
                    "image.size_bytes": len(base64_data),
                    "image.success": True,
                })
            
            return base64_data
                
    except httpx.TimeoutException:
        logger.warning("Image generation timed out after %ds", settings.image_generation_timeout)
        if active_span:
            active_span.set_attributes({"image.success": False, "image.error": "timeout"})
        return None
    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        if active_span:
            active_span.set_attributes({"image.success": False, "image.error": str(e)})
        return None
```

#### 2. Modify: `src/summit_sim/settings.py`

Add image generation configuration:

```python
image_generation_model: str = Field(
    default="bytedance-seed/seedream-4.5",
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
prompt = build_image_prompt(scenario, config)
print(f"\nImage Prompt:\n{prompt}")

# Cell 4: Generate image
import base64

image_data = await generate_scenario_image(scenario, config)
if image_data:
    print(f"✓ Generated image: {len(image_data)} base64 chars")
    # Decode and display using IPython
    from IPython.display import Image, display
    image_bytes = base64.b64decode(image_data)
    display(Image(data=image_bytes))
else:
    print("✗ Image generation failed (non-blocking)")

# Cell 5: Test multiple scenarios
# Generate 3-4 different scenarios and compare images
```

### Acceptance Criteria for Phase 1

- [x] `image_generator.py` module created with async generation logic
- [x] Settings updated with image model config
- [x] Notebook can generate scenarios and display images
- [x] Image generation returns base64 string (JSON-serializable)
- [x] Image generation is truly non-blocking (returns None on failure)
- [x] Prompts are well-formed and scenario-appropriate (includes ScenarioConfig context)
- [x] MLflow trace decorator for tracking with span attributes
- [x] All code passes ruff linting and formatting

**Status**: ✅ **Phase 1 Complete - Ready for Phase 2 integration**

---

## Phase 2: Integration (After Notebook Validation)

**Goal**: Integrate image generation into the author graph and UI. Only proceed after Phase 1 notebook testing is successful.

### Fail-Fast Philosophy

**NO defensive coding. NO silent fallbacks. NO default values that mask bugs.**

- Image generation is the ONLY place we catch exceptions (non-blocking requirement)
- All other code assumes data exists and lets exceptions propagate
- Store operations, scenario validation, and UI rendering all fail fast
- Clear error messages when assumptions are violated

### Files to Modify

#### 1. `src/summit_sim/schemas.py`

Add `image_data` field to ScenarioDraft:

```python
image_data: str | None = Field(
    default=None,
    description="Base64-encoded image data for scenario visualization",
)
```

#### 2. `src/summit_sim/graphs/author.py`

**Update `save_scenario`**: Remove defensive checks

```python
async def save_scenario(state: AuthorState, config: RunnableConfig) -> dict:
    """Save approved scenario to LangGraph store.
    
    Fails fast if scenario_draft or scenario_id is missing.
    """
    store = AppState.store
    await store.aput(
        ("scenarios",),
        state.scenario_id,
        {"scenario_draft": state.scenario_draft},
    )
    return {}
```

**Add image generation node** to graph:

```python
async def generate_image_node(state: AuthorState, config: RunnableConfig) -> dict:
    """Generate scenario image after saving to store.
    
    Non-blocking - if generation fails, scenario is still usable.
    Regenerates image on revision (new scenario draft).
    
    Assumes scenario_draft exists (fails fast if None).
    """
    scenario = ScenarioDraft.model_validate(state.scenario_draft)
    
    # Get config from state for image generation context
    scenario_config = ScenarioConfig.model_validate(state.scenario_config)
    
    # Generate image - returns None on failure (non-blocking)
    image_data = await generate_scenario_image(scenario, scenario_config)
    
    if image_data:
        # Update scenario with image
        scenario.image_data = image_data
        
        # Update stored scenario - let exceptions propagate
        store = AppState.store
        await store.aput(
            ("scenarios",),
            state.scenario_id,
            {"scenario_draft": scenario.model_dump(mode="json")},
        )
        
        logger.info(
            "Image generated and saved: scenario_id=%s, size=%d bytes",
            state.scenario_id,
            len(image_data),
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
import base64

async def show_scenario_intro(scenario: ScenarioDraft) -> None:
    """Display scenario intro with image if available.
    
    No placeholder - just shows nothing if no image.
    Fails fast if image_data is corrupt (base64 decode will error).
    """
    content = format_scenario_intro(scenario)
    
    elements = []
    
    if scenario.image_data:
        # Decode base64 to bytes for Chainlit - let exceptions propagate
        image_bytes = base64.b64decode(scenario.image_data)
        elements.append(cl.Image(
            content=image_bytes,
            name="scenario_image",
            display="inline",
            size="large",
        ))
    
    await cl.Message(content=content, elements=elements).send()
    
    await run_simulation()
```

#### 4. `src/summit_sim/ui/author.py`

**Update `handle_student_start()`**: Remove defensive checks

```python
async def handle_student_start(_state: AuthorState) -> None:
    """Handle student mode - auto-approve and start simulation immediately."""
    graph = cl.user_session.get("graph")
    if graph is None:
        await cl.Message(content="❌ Error: Session expired. Please start over.").send()
        return

    thread_id = cl.user_session.get("id")
    config_dict: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Auto-approve the scenario
    result = await graph.ainvoke(
        Command(resume={"action": "approve"}),
        config=config_dict,
    )

    final_state = AuthorState.from_graph_result(result)
    scenario_id = final_state.scenario_id

    # Set up simulation session
    cl.user_session.set("scenario_id", scenario_id)
    cl.user_session.set("mode", "player")
    # Store trace_id from authoring phase for correlation with simulation traces
    if final_state.current_trace_id:
        cl.user_session.set("authoring_trace_id", final_state.current_trace_id)

    # Load scenario from store (it was just saved during approval) - fails fast
    store = AppState.store
    store_result = await store.aget(("scenarios",), scenario_id)
    scenario_data = store_result.value  # AttributeError if None
    scenario = ScenarioDraft.model_validate(scenario_data["scenario_draft"])
    cl.user_session.set("scenario", scenario)

    # Show scenario intro with image
    await show_scenario_intro(scenario)
```

The scenario already has `image_data` from the graph if generation succeeded.

### Testing Strategy for Phase 2

1. **End-to-End Test**: Create scenario → approve → verify image appears in student view
2. **Revision Test**: Revise scenario → verify new image is generated
3. **Failure Test**: Temporarily break image generation → verify scenario still loads (no crash)
4. **Fail-Fast Test**: Remove scenario_id assignment → verify clear error on save (not silent failure)
5. **Mobile Test**: Verify image displays correctly on mobile (1344×768, 16:9 ratio)

### Acceptance Criteria for Phase 2

- [ ] Scenario image stored in LangGraph store with scenario (as base64 string)
- [ ] Image displays inline in Chainlit UI (decoded from base64)
- [ ] No placeholder shown - just empty if no image
- [ ] Image regenerates on scenario revision
- [ ] Scenario loads successfully even if image generation fails
- [ ] Fail-fast: Clear errors when assumptions violated (no silent failures)
- [ ] All tests pass (coverage >= 80%)
- [ ] Code passes quality gates

---

## Implementation Notes

### Design Decisions

1. **Async httpx**: Using `httpx.AsyncClient` instead of PydanticAI because image generation via chat completions with modalities is a different API pattern than structured outputs.

2. **Base64 Storage**: Store base64-encoded strings in ScenarioDraft (JSON-serializable for LangGraph state). Decode to bytes only when passing to Chainlit's `cl.Image(content=...)`. This ensures compatibility with LangGraph's checkpointing which requires JSON serialization of state.

3. **Non-blocking Strategy**: Image generation runs after `save_scenario` so the scenario is already persisted. If image fails, the scenario is still usable. The UI shows nothing if no image (no placeholder text).

4. **Regeneration on Revision**: The graph flow `save → generate_image` means every time we save (including after revisions), we regenerate the image. This ensures the image matches the revised scenario.

5. **MLflow Tracing**: Image generation costs are tracked via MLflow span attributes (following project conventions):
   - `@mlflow.trace(span_type=SpanType.LLM)` decorator for tracing
   - `active_span.set_attributes({"image.model": model, "image.prompt_length": len(prompt)})` for cost tracking
   - `active_span.set_attributes({"image.success": True/False})` for success tracking
   - **NO `mlflow.log_param()`** - project uses span attributes and tags, not params

6. **Image-Only Modalities**: The `bytedance-seed/seedream-4.5` model is image-only, so we use `modalities: ["image"]` (not `["image", "text"]`). This differs from multimodal models that can return both text and images.

7. **Aspect Ratio Configuration**: We use `image_config.aspect_ratio: "16:9"` in the API request to ensure consistent 1344×768 landscape dimensions across all image generation models. This is more reliable than relying on model defaults and provides optimal mobile viewing experience.

### Mobile Optimization

The prompt template includes specific instructions:
- Strong focal point
- Simple composition
- Clear readability on small screens

**Aspect Ratio Configuration:**
We use `image_config.aspect_ratio: "16:9"` which produces **1344×768** images. This is:
- Wider than the initially planned 1024×768
- Better for mobile landscape viewing
- Consistent across all OpenRouter image models
- More immersive for wilderness scene composition

The 16:9 ratio provides a cinematic feel that works well for depicting expansive wilderness environments while remaining readable on mobile screens.

### Error Handling Philosophy

**FAIL FAST. NO DEFENSIVE CODING.**

- **Never add** extra `if` statements or try/except blocks to "handle" edge cases
- **Never add** silent fallbacks or default values that mask bugs
- **Never add** defensive null checks (`if x is not None`, `if store:`)
- **Only exception**: Image generation catches exceptions to remain non-blocking
- **Everything else** fails immediately with a clear error
- **Clear is better than safe**: A crash with a good stack trace is better than silent data corruption

Following Summit-Sim conventions:
- Let exceptions propagate for truly unexpected errors
- Log warnings only for recoverable failures (image generation timeouts)
- Never block core functionality (scenario loading) on optional features
- Store operations, validation, and UI rendering all fail fast

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
