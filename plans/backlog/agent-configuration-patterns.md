# Agent Configuration Patterns

This document outlines recommended patterns for managing agent configuration and instantiation in Summit Sim. These approaches allow per-agent model selection and reduce boilerplate.

## Pattern 1: Module-Level Singletons (Current - Simplest)

**Best for:** Hackathons, rapid prototyping, simple deployments

Instantiate agents once at module import time. No config files, minimal boilerplate.

```python
# agents/generator.py
_GENERATOR_AGENT = Agent(
    model=OpenRouterModel("nvidia/nemotron-3-super-120b-a12b:free", provider=_PROVIDER),
    output_type=ScenarioDraft,
    system_prompt=GENERATOR_SYSTEM_PROMPT,
    model_settings=OpenRouterModelSettings(openrouter_reasoning={"effort": "high"}),
)

async def generate_scenario(host_config: HostConfig) -> ScenarioDraft:
    result = await _GENERATOR_AGENT.run(prompt)
    return result.output
```

### Pros
- Zero config files to manage
- Zero boilerplate in functions
- Agents instantiated exactly once
- Simplest mental model

### Cons
- Models hardcoded (requires code change to swap)
- No per-environment configuration
- Difficult to mock for testing

---

## Pattern 2: Dataclass Configuration (Recommended for Growth)

**Best for:** Production applications, when you need type safety and testability

Define agent configurations as dataclasses, cache agent instances.

### Implementation

```python
# agents/config.py
from dataclasses import dataclass
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from functools import lru_cache

from summit_sim.settings import settings

@dataclass(frozen=True)
class AgentConfig:
    """Configuration for an agent.
    
    Attributes:
        model: Model identifier (e.g., "nvidia/nemotron-3-super-120b-a12b:free")
        reasoning_effort: Reasoning level ("low", "medium", "high")
        system_prompt: System prompt for the agent
        output_type: Pydantic model for structured output
    """
    model: str
    reasoning_effort: str
    system_prompt: str
    output_type: type

# Per-agent configurations
AGENTS = {
    "generator": AgentConfig(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        reasoning_effort="high",
        system_prompt=GENERATOR_SYSTEM_PROMPT,  # Import from agent module
        output_type=ScenarioDraft,
    ),
    "simulation_feedback": AgentConfig(
        model="anthropic/claude-3.5-sonnet",
        reasoning_effort="medium",
        system_prompt=FEEDBACK_SYSTEM_PROMPT,
        output_type=SimulationResult,
    ),
    "safety_judge": AgentConfig(
        model="anthropic/claude-3.5-sonnet",
        reasoning_effort="high",
        system_prompt=SAFETY_JUDGE_PROMPT,
        output_type=JudgeResult,
    ),
    "realism_judge": AgentConfig(...),
    "pedagogy_judge": AgentConfig(...),
    "refiner": AgentConfig(...),
    "debrief": AgentConfig(...),
}

# Shared provider instance
_provider = OpenRouterProvider(api_key=settings.openrouter_api_key)

@lru_cache(maxsize=None)
def get_agent(agent_name: str) -> Agent:
    """Get or create an agent by name.
    
    Uses lru_cache to ensure each agent is instantiated exactly once.
    
    Args:
        agent_name: Name of the agent (key in AGENTS dict)
        
    Returns:
        Configured Agent instance
        
    Raises:
        KeyError: If agent_name is not found in AGENTS
        
    Example:
        >>> agent = get_agent("generator")
        >>> result = await agent.run(prompt)
    """
    if agent_name not in AGENTS:
        raise KeyError(f"Unknown agent: {agent_name}. Available: {list(AGENTS.keys())}")
    
    cfg = AGENTS[agent_name]
    
    model = OpenRouterModel(
        cfg.model,
        provider=_provider,
    )
    
    model_settings = OpenRouterModelSettings(
        openrouter_reasoning={"effort": cfg.reasoning_effort}
    )
    
    return Agent(
        model=model,
        output_type=cfg.output_type,
        system_prompt=cfg.system_prompt,
        model_settings=model_settings,
    )
```

### Usage

```python
# In your agent functions
from summit_sim.agents.config import get_agent

async def generate_scenario(host_config: HostConfig) -> ScenarioDraft:
    agent = get_agent("generator")
    result = await agent.run(prompt)
    return result.output
```

### Pros
- Type-safe configurations
- IDE autocomplete support
- Easy to modify without YAML complexity
- Agents cached (instantiated once)
- Testable: can mock get_agent or inject test configs

### Cons
- Config is in code (requires redeploy to change models)
- All agents must be defined at startup

---

## Pattern 3: YAML-Based Registry (Most Flexible)

**Best for:** Large teams, A/B testing, environment-specific model routing

Externalize configuration to YAML for maximum flexibility without code changes.

### Implementation

```python
# agents/registry.py
import yaml
from typing import Any
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

from summit_sim.settings import settings

class AgentRegistry:
    """Registry for managing agent configurations and instances.
    
    Loads agent configurations from YAML and provides lazy initialization
    with caching. Supports singleton pattern and Chainlit session storage.
    
    Attributes:
        config_path: Path to YAML configuration file
        _agents: Cache of instantiated agents
        _config: Loaded configuration dict
        _provider: Shared OpenRouter provider instance
    
    Example:
        >>> registry = AgentRegistry("config/models.yaml")
        >>> agent = registry.get("generator")
        >>> result = await agent.run(prompt)
    """
    
    def __init__(self, config_path: str):
        """Initialize registry with YAML config.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self._agents: dict[str, Agent] = {}
        self._config = self._load_config()
        self._provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    
    def _load_config(self) -> dict[str, Any]:
        """Load and validate YAML configuration.
        
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def get(self, agent_name: str) -> Agent:
        """Get or create an agent by name.
        
        Lazy initialization - creates agent on first access and caches it.
        
        Args:
            agent_name: Name of the agent (key in YAML config)
            
        Returns:
            Configured Agent instance
            
        Raises:
            KeyError: If agent_name not found in config
            ValueError: If agent configuration is invalid
            
        Example:
            >>> agent = registry.get("generator")
            >>> result = await agent.run("Generate a scenario...")
        """
        if agent_name not in self._agents:
            if agent_name not in self._config.get("agents", {}):
                available = list(self._config.get("agents", {}).keys())
                raise KeyError(
                    f"Agent '{agent_name}' not found in {self.config_path}. "
                    f"Available: {available}"
                )
            
            cfg = self._config["agents"][agent_name]
            self._agents[agent_name] = self._create_agent(cfg)
        
        return self._agents[agent_name]
    
    def _create_agent(self, cfg: dict[str, Any]) -> Agent:
        """Create an agent from configuration dictionary.
        
        Args:
            cfg: Agent configuration dict with keys:
                - model: str (required)
                - reasoning_effort: str (default: "medium")
                - system_prompt: str (required)
                - output_type: str (required, must be importable)
                
        Returns:
            Configured Agent instance
        """
        model = OpenRouterModel(cfg["model"], provider=self._provider)
        
        model_settings = OpenRouterModelSettings(
            openrouter_reasoning={"effort": cfg.get("reasoning_effort", "medium")}
        )
        
        # Dynamic import of output_type
        output_type = self._import_type(cfg["output_type"])
        
        return Agent(
            model=model,
            output_type=output_type,
            system_prompt=cfg["system_prompt"],
            model_settings=model_settings,
        )
    
    def _import_type(self, type_path: str) -> type:
        """Dynamically import a type from module path.
        
        Args:
            type_path: Full module path (e.g., "summit_sim.schemas.ScenarioDraft")
            
        Returns:
            Imported type/class
        """
        module_path, class_name = type_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    
    def reload(self) -> None:
        """Reload configuration from disk and clear agent cache.
        
        Useful for hot-reloading config without restarting the app.
        """
        self._config = self._load_config()
        self._agents.clear()
```

### Configuration File

```yaml
# config/models.yaml
defaults:
  reasoning_effort: medium
  provider: openrouter

agents:
  generator:
    model: nvidia/nemotron-3-super-120b-a12b:free
    reasoning_effort: high
    system_prompt: |
      You are an expert wilderness rescue scenario designer...
    output_type: summit_sim.schemas.ScenarioDraft
    
  simulation_feedback:
    model: anthropic/claude-3.5-sonnet
    reasoning_effort: medium
    system_prompt: |
      You are a wilderness rescue instructor providing personalized feedback...
    output_type: summit_sim.schemas.SimulationResult
    
  safety_judge:
    model: anthropic/claude-3-opus
    reasoning_effort: high
    system_prompt: |
      You are a medical accuracy validator...
    output_type: summit_sim.schemas.JudgeResult
    
  realism_judge:
    model: anthropic/claude-3.5-sonnet
    reasoning_effort: medium
    system_prompt: ...
    output_type: summit_sim.schemas.JudgeResult
    
  pedagogy_judge:
    model: anthropic/claude-3.5-sonnet
    reasoning_effort: medium
    system_prompt: ...
    output_type: summit_sim.schemas.JudgeResult
    
  refiner:
    model: anthropic/claude-3.5-sonnet
    reasoning_effort: high
    system_prompt: ...
    output_type: summit_sim.schemas.ScenarioDraft
    
  debrief:
    model: anthropic/claude-3.5-sonnet
    reasoning_effort: medium
    system_prompt: ...
    output_type: summit_sim.schemas.DebriefReport
```

### Chainlit Integration

```python
# app.py or main entry point
import chainlit as cl
from summit_sim.agents.registry import AgentRegistry

@cl.on_chat_start
async def on_chat_start():
    """Initialize registry in session on first user connection."""
    # Module-level singleton (one per process)
    registry = AgentRegistry("config/models.yaml")
    cl.user_session.set("agent_registry", registry)

# Usage in handlers
async def generate_scenario(host_config: HostConfig) -> ScenarioDraft:
    registry = cl.user_session.get("agent_registry")
    agent = registry.get("generator")
    result = await agent.run(prompt)
    return result.output
```

### Alternative: True Singleton

For true singleton (one registry per process, shared across all sessions):

```python
# agents/registry.py
import threading

_registry: AgentRegistry | None = None
_lock = threading.Lock()

def get_registry(config_path: str = "config/models.yaml") -> AgentRegistry:
    """Get or create singleton registry instance.
    
    Thread-safe singleton pattern for process-wide registry.
    
    Args:
        config_path: Path to YAML config (only used on first call)
        
    Returns:
        Singleton AgentRegistry instance
    """
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = AgentRegistry(config_path)
    return _registry
```

### Pros
- No code changes needed to swap models
- Supports A/B testing via config changes
- Environment-specific configs (dev/staging/prod)
- Hot-reloadable without restart
- Non-technical team members can modify models

### Cons
- More complexity than dataclass approach
- Runtime errors if YAML is malformed
- Dynamic imports add indirection
- Requires YAML parsing dependency

---

## Decision Matrix

| Pattern | Setup Complexity | Flexibility | Type Safety | Best For |
|---------|-----------------|-------------|-------------|----------|
| Module Singleton | ⭐ Lowest | Low | ✅ High | Hackathons, simple apps |
| Dataclass Config | ⭐⭐ Low | Medium | ✅✅ Very High | Production apps, teams |
| YAML Registry | ⭐⭐⭐ Medium | ✅✅ Very High | Medium | Large teams, A/B testing |

## Migration Path

If starting with Pattern 1 and want to migrate:

1. **1 → 2**: Move agent configs from inline to dataclass, use `get_agent()` wrapper
2. **2 → 3**: Add YAML loading layer on top of dataclass configs, maintain backwards compatibility

The dataclass pattern (Pattern 2) is the sweet spot for most applications - type-safe, testable, and simple without the YAML overhead.
