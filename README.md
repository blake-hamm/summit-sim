# 🏔️ Summit-Sim

**An AI-powered wilderness rescue simulator.** 

Summit-Sim uses human-in-the-loop review to generate curriculum-informed, interactive backcountry emergencies for dynamic Wilderness First Responder (WFR) training.

<img src="public/favicon.png" alt="summit-sim" width="200"/>

## 💡 The Problem
Wilderness First Responder (WFR) training relies heavily on static paper scenarios or expensive live-action roleplay. Students rarely get enough dynamic, unpredictable repetitions to truly test their decision-making under pressure. When the unexpected happens in the backcountry, textbook memorization isn't enough—responders need dynamic critical thinking.

## 🚀 The Solution
Summit-Sim provides infinite, medically accurate WFR scenarios through a dynamic AI game loop. Instead of multiple-choice questions, responders use natural language to evaluate scenes, check vitals, and apply treatments. The system evaluates every action against a hidden medical "truth," evolving the scenario exactly as a real patient's condition would change in the wild.

### Key Features
*   **Infinite Scenario Authoring:** Generate highly specific emergencies based on environment, group size, and difficulty level.
*   **Dynamic State Machine:** The patient's condition evolves realistically based on the responder's timeline and medical interventions (or lack thereof).
*   **Human-In-The-Loop (HITL) Validation:** Instructors review and approve scenarios before publication, with all feedback logged to MLflow for continuous improvement.
*   **Objective Debriefing:** Post-simulation evaluation scores the responder's actions against established WFR protocols.

---

## 🏗️ Technical Architecture 

Summit-Sim is built with a sophisticated, production-ready AI stack focused on latency, strict output schemas, and agentic orchestration. 

### Core Tech Stack
*   **Orchestration (LangGraph):** Manages the complex state flows of the simulation. Utilizes `StateGraph` for the core loops, `interrupt()` for instructor HITL injections, and checkpointing for state persistence.
*   **Agent Framework (PydanticAI):** Ensures absolute medical safety and system stability by strictly enforcing all LLM inputs and outputs via Pydantic `BaseModel` schemas.
*   **Frontend UI (Chainlit):** A fully asynchronous, reactive Python UI tailored for conversational AI, providing seamless UX for both scenario authoring and active gameplay.
*   **Observability (MLflow):** Comprehensive LLM span tracing, variable logging, and feedback tracking to monitor agent reasoning and model performance.

### System Flow
The application is cleanly divided into two interconnected graphs joined by a shared Scenario ID:

1.  **The Authoring Graph:** Takes environmental parameters and dynamically generates the baseline blueprint (Setting, Patient vitals, Hidden Medical Truth, and Turn 0). Instructors review via HITL interrupt before scenarios go live.
2.  **The Simulation Graph:** A continuous game loop where the AI evaluates open-ended student actions against the hidden truth, dynamically updating the active scene state and generating the next narrative frame.

### Agent Architecture
Three specialized PydanticAI agents power the system:
- **Generator:** Creates wilderness rescue scenarios from minimal configuration
- **Action Responder:** Evaluates student free-text actions and provides cumulative scoring (0-100%)
- **Debrief:** Generates post-simulation performance analysis against WFR protocols

*Planned: MLflow automatic validation judges (Safety, Realism, Pedagogy) for medical accuracy assessment.*

---

*Built for the Weber State AI Hackathon* 🐾


#### Resources:

- https://www.scribd.com/document/38484292/Wilderness-First-Responder-Course
- https://wildsafe.org/wp-content/uploads/2021/11/CWS-%E2%80%93-WFR-Partcipant-Workbook-2021.pdf
- https://www.nols.edu/category/wilderness-medicine/case-studies/
- https://www.nols.edu/category/wilderness-medicine/case-studies/page/2/
- https://www.letsgoexploring.com/public/wilderness-first-responder-infosheet-side-A-2022.pdf
- https://www.wildmedcenter.com/uploads/5/9/8/2/5982510/standard_wfr_syllabus.pdf
- https://sierrarescue.com/wp-content/uploads/2013/12/WILDERNESS-FIRST-RESPONDER-General-Info.pdf
- https://sierrarescue.com/coursepdf/WFR.pdf
- https://www.nols.edu/wp-content/uploads/2025/08/23732-Student-Logistics.pdf
- https://www.deepsprings.edu/next/wp-content/uploads/2024/07/Student-WFR-Training.pdf
