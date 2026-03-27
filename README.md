# summit-sim

Summit-Sim is an AI wilderness rescue simulator. It uses multi-agent validation to generate medically safe, interactive backcountry emergencies for dynamic first-responder training.

<img src="public/favicon.png" alt="summit-sim" width="200"/>

## Architecture & Naming Conventions

To support both self-serve (student) practice and instructor-led sessions, this project uses functional naming rather than persona-based naming. This ensures our codebase accurately reflects system behavior, even when a single user fulfills multiple roles across the application.

### Core Terminology
We explicitly separate the static blueprint from the active gameplay session:
*   **Scenario:** The static blueprint or configuration. This includes the setting, patient background, the hidden medical truth, and the initial state. 
*   **Simulation:** The active, stateful playthrough of a Scenario. A simulation is instantiated from a scenario.

### UI/UX Roles (The "Who")
Instead of rigid "Teacher" and "Student" designations, UI views and permissions are mapped to functional roles. A single user might act as both the Author and the Player in a self-serve session.

*   **The Author (formerly Teacher):** The user setting up the environment. They define the parameters (participants, activity, difficulty) to generate a Scenario.
*   **The Player / Responder (formerly Student):** The user actively engaged in the Simulation. They input free-text actions to evaluate, treat, and respond to the medical emergency.
*   **The Reviewer (HITL):** An optional, privileged view (often used by instructors) that can observe the hidden truth of a Simulation, interrupt the state graph, and inject dynamic feedback or complications.

### Backend & Graph State Naming (The "How")
The backend is split into two distinct LangGraph workflows joined by a shared `scenario_id`. 

#### 1. The Authoring Graph (Blueprint Generation)
*Responsible for taking initial parameters and generating the `ScenarioBase`.*
*   **File/Module Name:** `authoring.py` or `generator.py`
*   **Graph Name:** `AuthoringGraph`
*   **Key Schemas:**
    *   `ScenarioConfig` (Replaces `TeacherConfig`): The initial form inputs (e.g., environment, difficulty).
    *   `ScenarioBlueprint` / `ScenarioBase`: The structured output containing the Setting, Patient state, Hidden Truth, and Turn 0 narrative.
    *   `ReviewerAdjustment`: Human-in-the-loop (HITL) injections used to rebuild or refine scenario.

#### 2. The Simulation Graph (Dynamic Game Loop)
*Responsible for handling continuous free-text input, updating the medical state, and returning the next narrative frame.*
*   **File/Module Name:** `simulation.py`
*   **Graph Name:** `SimulationGraph`
*   **Key Schemas:**
    *   `PlayerAction` (Replaces `StudentAction`): The free-text input submitted by the user.
    *   `SimulationTurnResult`: The PydanticAI model that evaluates the action against the blueprint's hidden truth, updates the active scene state, and generates the narrative response.
    *   `SimulationState`: The overarching LangGraph state holding the chat history, current patient vitals, and active scene status.

#### 3. The Evaluation Graph (Post-Simulation)
*Responsible for concluding the session and grading the user's performance.*
*   **File/Module Name:** `debrief.py`
*   **Key Schemas:**
    *   `ActionLog`: The compiled history of `PlayerAction`s.
    *   `DebriefReport`: The final output evaluating the medical accuracy of the runtime decisions against the WFR protocol.


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
