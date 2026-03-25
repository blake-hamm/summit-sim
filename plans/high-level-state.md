# State Management Plan

## Overview

This application has three separate state layers: Chainlit manages live UI session state, LangGraph manages workflow execution state, and MLflow manages observability, traces, evaluation runs, and human feedback records. The key design rule is that feedback belongs on the exact MLflow trace being evaluated, not only on the broader run that contains that trace. [mlflow](https://mlflow.org/docs/latest/api_reference/_modules/mlflow/entities/assessment.html)

The teacher flow and student flow can remain separate LangGraph graphs and separate state objects, as long as they share a common application-level identifier for the same simulation lineage. This keeps graph logic clean while still letting MLflow correlate the full teacher-to-student lifecycle through shared tags and trace search.

## Identifier Model

| Identifier | System of record | Meaning | Where to store it |
|---|---|---|---|
| `simulation_id` | Application backend | Canonical ID for one teacher-authored simulation and all related student playthroughs  [learn.microsoft](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-conversations). | Store in teacher state, student state, and as an MLflow tag on every related run or trace  [learn.microsoft](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/evaluate-conversations). |
| `mlflow_run_id` | MLflow | Top-level evaluation container that can hold one or more traces for a bounded execution context  [mlflow](https://mlflow.org/blog/multiturn-evaluation). | Save in LangGraph state only for correlation, and use it inside MLflow for navigation and grouping  [mlflow](https://mlflow.org/blog/multiturn-evaluation). |
| `trace_id` | MLflow | Identifier for the exact generation, student-step, or debrief trace that receives feedback  [mlflow](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/). | Pass through callback context or message metadata whenever the UI needs to submit feedback on a specific response  [docs.databricks](https://docs.databricks.com/api/workspace/mlflowexperimenttrace/createassessmentv3). |
| `thread_id` | LangGraph | Persistence key for one isolated graph execution state, especially important when multiple students may work the same simulation concurrently  [community.latenode](https://community.latenode.com/t/how-to-preserve-state-and-resume-workflows-in-langchain-with-human-intervention/39108). | Use one teacher `thread_id` for the teacher graph and one distinct student `thread_id` per student playthrough  [community.latenode](https://community.latenode.com/t/how-to-preserve-state-and-resume-workflows-in-langchain-with-human-intervention/39108). |
| `chainlit_session_id` | Chainlit | UI session identifier for a live chat/browser session  [github](https://github.com/Chainlit/chainlit/issues/1385). | Store in student state for correlation and log it into MLflow as a tag, but do not use it as the global simulation key  [mlflow](https://mlflow.org/docs/latest/genai/tracing/search-traces/). |

## System Responsibilities

| System | State it owns | IDs it should know | IDs it should not own as primary keys |
|---|---|---|---|
| Chainlit | Active chat session, current user interaction context, and UI-triggered feedback actions  [github](https://github.com/Chainlit/chainlit/issues/1385). | `chainlit_session_id`, current `thread_id`, current `trace_id`, and `simulation_id` for correlation  [docs.databricks](https://docs.databricks.com/api/workspace/mlflowexperimenttrace/createassessmentv3). | It should not define the canonical simulation lineage or MLflow run structure  [mlflow](https://mlflow.org/blog/multiturn-evaluation). |
| LangGraph | Teacher graph state, student graph state, interrupts, and persistent execution memory  [community.latenode](https://community.latenode.com/t/how-to-preserve-state-and-resume-workflows-in-langchain-with-human-intervention/39108). | `simulation_id`, `thread_id`, optional `mlflow_run_id`, and any current trace metadata needed for callbacks  [mlflow](https://mlflow.org/blog/multiturn-evaluation). | It should not be the system of record for feedback history, because feedback belongs in MLflow assessments  [mlflow](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/). |
| MLflow | Runs, traces, session-style tags, and human feedback assessments  [mlflow](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/). | `mlflow_run_id`, `trace_id`, `simulation_id` tag, and `chainlit_session_id` tag when applicable  [docs.databricks](https://docs.databricks.com/api/workspace/mlflowexperimenttrace/createassessmentv3). | It does not need to own LangGraph `thread_id` semantics; if needed, record `thread_id` only as a searchable tag  [mlflow](https://mlflow.org/docs/latest/genai/tracing/search-traces/). |

## Recommended Flow

1. When the teacher starts a simulation, the backend creates a new `simulation_id`, the teacher graph starts with its own `thread_id`, and MLflow starts a run for that teacher authoring session with the `simulation_id` attached as a tag. [mlflow](https://mlflow.org/blog/multiturn-evaluation)
2. When the teacher reviews or edits the generated simulation, any approval or correction should be logged as MLflow feedback on the exact teacher trace that produced the content being reviewed. [docs.databricks](https://docs.databricks.com/api/workspace/mlflowexperimenttrace/createassessmentv3)
3. When a student begins a playthrough, create a new student `thread_id`, keep the same `simulation_id`, capture the `chainlit_session_id`, and start a separate MLflow run for that student execution so the playthrough remains isolated from other students while still queryable by shared tags. [github](https://github.com/Chainlit/chainlit/issues/1385)
4. For each student step and the final debrief, log SME or student feedback to the exact `trace_id` that generated that response, and include `simulation_id` plus `chainlit_session_id` as tags for later search and dataset building. [mlflow](https://mlflow.org/docs/latest/genai/tracing/search-traces/)

## Operating Rules

Use `simulation_id` as the single cross-system business key for the full scenario lifecycle, because it survives teacher authoring, student execution, and later dataset extraction better than a UI-specific session ID. Use `thread_id` to isolate LangGraph memory for each live execution, especially once you move from in-memory state to Redis-backed persistence. [community.latenode](https://community.latenode.com/t/how-to-preserve-state-and-resume-workflows-in-langchain-with-human-intervention/39108)

Use `chainlit_session_id` as a UI correlation tag, not as the canonical identifier for the simulation itself. Use `mlflow_run_id` for run-level organization, but use `trace_id` for feedback collection, because MLflow assessments are attached at the trace level. [mlflow](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/)

If you want, I can turn this into a tighter implementation checklist with exact field names such as `simulation_id`, `student_thread_id`, `chainlit_session_id`, `mlflow_run_id`, and `trace_id`.