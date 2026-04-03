# 🏔️ Summit-Sim

**An AI-powered wilderness rescue simulator with human-in-the-loop validation.**

https://summit-sim.bhamm-lab.com/

Summit-Sim generates curriculum-informed, interactive backcountry emergencies for Wilderness First Responder (WFR) training. Instructors review and approve AI-generated scenarios before students engage with them, ensuring medical accuracy and pedagogical value.

<img src="public/favicon.png" alt="summit-sim" width="200"/>

*Built for the [Weber State AI Hackathon](https://hackathon.weber.edu/)* 🐾

**Acknowledgments:** Thanks to **Keenan Grady** at the [Ogden Avalanche Center](https://ogdenavalanche.org/) for domain expertise in wilderness safety and WFR curriculum.

## 💡 The Problem

Wilderness First Responder (WFR) training relies heavily on static paper scenarios or expensive live-action roleplay. Students rarely get enough dynamic, unpredictable repetitions to truly test their decision-making under pressure. When emergencies happen in the backcountry, textbook memorization isn't enough—responders need dynamic critical thinking developed through varied practice.

## 🚀 The Solution

Summit-Sim provides infinite, medically accurate WFR scenarios through a dynamic AI game loop. Instead of multiple-choice questions, students use natural language to evaluate scenes, check vitals, and apply treatments. The system evaluates every action against a hidden medical "truth," evolving the scenario exactly as a real patient's condition would change.

### Key Features

*   **Infinite Scenario Generation:** Create highly specific emergencies based on environment, group size, difficulty, and complexity. Each scenario includes a unique AI-generated atmospheric image.
*   **Human-In-The-Loop (HITL) Validation:** Instructors review scenarios before publication, with all feedback logged to MLflow for continuous improvement.
*   **Natural Language Interaction:** Students type free-text actions (no multiple choice). The AI evaluates against WFR protocols and reveals information progressively.
*   **Cumulative PAS Scoring:** Tracks student progress (0-100%) across 5 Patient Assessment System milestones: Scene Size-up, Primary Assessment, Secondary Assessment, Treatment, and Evacuation Plan.
*   **Intelligent Debriefing:** Post-simulation analysis with clinical reasoning assessment, key mistakes identification, and personalized recommendations.
*   **Shareable Scenarios:** Students join via unique URLs—multiple students can work on the same scenario in different sessions.
*   **Solo Practice Mode:** Students can also generate their own scenarios for independent practice. The scenario auto-generates and immediately starts the simulation without instructor review. **Students never see hidden information** (learning objectives, hidden truth, or hidden state)—they must discover medical details through proper assessment, just like in the instructor-led flow.

---

## 🏗️ Technical Architecture

Summit-Sim is built with a sophisticated AI stack focused on medical safety, strict output schemas, and comprehensive observability.

### Core Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Complex state flows, HITL interrupts via interrupt(), checkpointing for persistence |
| **Agent Framework** | PydanticAI | Strict medical safety via enforced Pydantic schemas for all LLM I/O |
| **UI Framework** | Chainlit | Async, reactive Python UI for conversational flows |
| **Observability** | MLflow | LLM span tracing, human feedback tracking, prompt versioning, GEPA optimization |
| **State Storage** | Redis (local) / DragonflyDB (prod) | LangGraph checkpoint persistence and scenario storage |
| **Image Generation** | OpenRouter (Gemini Flash Image) | Unique atmospheric scenario images (16:9, non-blocking) |

### System Flow

The application runs two interconnected LangGraph workflows:

1.  **The Authoring Graph:** Generates scenarios from environmental parameters, creates atmospheric images, and presents them for instructor review via HITL interrupt before publication. When students generate their own scenarios, this graph auto-approves and bypasses the review interrupt, jumping directly to simulation.
2.  **The Simulation Graph:** A continuous game loop where students type actions, the AI evaluates against hidden medical truth, and the patient state evolves dynamically. Completes at 80% PAS milestone completion or max turns. **All students** (whether joining via instructor link or generating solo) see only observable information—hidden medical data is never revealed without proper assessment.

### Agent Architecture

Four specialized PydanticAI agents power the system:

| Agent | Purpose | Output Schema |
|-------|---------|---------------|
| **Generator** | Creates wilderness rescue scenarios from configuration | ScenarioDraft with visible/hidden separation |
| **Image Generator** | Produces unique scenario images | Base64-encoded 16:9 image |
| **Action Responder** | Evaluates student actions, reveals information, updates PAS score | ActionResponse with narrative and feedback |
| **Debrief** | Generates post-simulation performance analysis | DebriefReport with teaching points |

### Validation and Optimization

**MLflow Judges (Implemented):** Four specialized judges evaluate simulation quality:
- **Structure Judge:** Score range validation, narrative length, feedback tone
- **Scoring Judge:** Milestone justification, monotonic progress, acknowledgment
- **Medical Judge:** Treatment gate validation (was_correct accuracy)
- **Continuity Judge:** Progressive revelation, score monotonicity

Note: Judges are implemented but currently disabled in production due to MLflow bug #20782. They are used for offline GEPA (GenAI Evaluation and Prompt Alignment) optimization in Jupyter notebooks.

**GEPA Optimization:** Uses the judge framework with reflection models to iteratively improve agent prompts based on expert feedback. See notebooks/action-responder-prompt.ipynb.

---

## 📚 Resources

WFR Curriculum references used to ensure medical accuracy:

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
