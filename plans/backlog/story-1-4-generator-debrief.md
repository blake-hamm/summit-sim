# Story Backlog: Generator Scenario Debrief

## Overview

Future story for analyzing scenario generation quality. This is intentionally lower priority than the student simulation debrief.

**Status**: BACKLOG (not planned for current sprint)  
**Priority**: Low  
**Dependencies**: Validation judges (Safety, Realism, Pedagogy)  

---

## Purpose

Analyze the quality of AI-generated scenarios before they reach students. This would provide:
- Medical accuracy scoring
- Realism assessment
- Pedagogical value metrics
- Comparison against golden scenarios

---

## When to Implement

Consider implementing when:
- Validation judges (Story 2.x) are implemented
- Need automated quality gates for scenario generation
- Collecting data for model A/B testing
- Building golden scenario dataset

---

## Potential Schema

```python
class GeneratorDebriefReport(BaseModel):
    """Analysis of scenario generation quality."""
    
    scenario_id: str
    generation_timestamp: datetime
    
    # Quality metrics
    medical_accuracy_score: float  # 0-100
    realism_score: float  # 0-100
    pedagogy_score: float  # 0-100
    overall_quality_score: float  # 0-100
    
    # Analysis
    medical_issues: list[str]
    realism_concerns: list[str]
    learning_objectives_met: list[str]
    suggested_improvements: list[str]
    
    # Comparison
    similarity_to_golden: float | None  # If comparable golden scenario exists
    
    # Decision
    approved_for_use: bool
    requires_refinement: bool
```

---

## Integration Points

- Would run after `generate_scenario()` completes
- Could be part of validation loop (pre-judges)
- Could replace/augment validation judges
- Logs to MLflow generation run

---

## Notes

- May be redundant with validation judges
- Could serve different purpose: judge the AI's generation capability vs judge the scenario content
- Useful for collecting training data on good/bad generations
- Could inform few-shot prompting improvements

---

## Related Stories

- Story 1.3: Student Simulation Debrief (current focus)
- Future: Validation judges (Safety, Realism, Pedagogy)
- Future: Golden scenario management
