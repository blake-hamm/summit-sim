### Show the main Chainlit UI
- For the Weber State Hackathon, I created this app called Summit-Sim.
- I collaborated with Ogden Avalanche Center President Keenan Grady for professional validation.
- Currently, Wilderness First Responder courses rely on pre-canned scenarios from textbooks.
- Summit-Sim creates infinitely many, AI-generated scenarios.
- It provides dynamic gameplay to prep students for their WFR exams.

### Click "Create Scenario," pick parameters, and submit
- While this builds, let me explain what's happening under the hood.
- Our LangGraph architecture is orchestrating multiple steps.
- Instead of just asking an LLM for a story, we use PydanticAI to force the model into strict data structures.
- The AI is simultaneously building the physical environment, baseline vitals, and the "Hidden Truth" patient state.

### Type "add a rattlesnake in the hidden state. do not reveal it in any other fields."
- We use LangGraph's interrupt feature for a Human-in-the-Loop experience.
- This lets instructors easily tweak the scenario before publishing.
- Approve is logging with agent traces to mlflow

### Open the shareable URL and open the Student View
- The student sees a clean interface where the hidden truth is completely concealed.
  
### Type "Assess surroundings for safety"
- In this multi-agent Simulation Graph, students use open-text input.
- The agent evaluates their real-time medical decisions against the hidden state and WFR curriculum, dynamically updating the scene.

### Switch to a tab with a pre-completed scenario
- To save time, here is a completed scenario.
- Once the student finishes, the final LangGraph node generates a comprehensive debrief report.
- We also track all LLM spans and feedback via MLflow for full observability. 

### Conclusion
- Our goal was to hide sophisticated AI orchestration behind an intuitive UI.
- Summit-Sim delivers a unique training tool to prep students for the WFR exam or brush up on stale skills.
