### Show the main Chainlit UI
- Currently, Wilderness First Responder courses rely on pre-canned scenarios from textbooks.
- To solve this, I collaborated with the Ogden Avalanche Center President to build Summit-Sim—an AI-generated simulator that creates infinitely many dynamic medical emergencies to prepare students for exam

### Click "Create Scenario," pick parameters, and submit
- The instructor sets the parameters, and our generator agent creates a WFR-aligned scenario.
- With multimodal AI, it generates a unique image per scenario
- Here are all the other ouptus, leveraging pydantic AI to enforce strict input/output types

### Type "add a rattlesnake in the hidden state. do not reveal it in any other fields."
- Using langgraph interupt, Teacher is able to revise the scenario, for example 'add a rattlesnake'
- Hidden information is presented only to the teacher and will be dynamically revealed in the action agent based on student actions

### Open the shareable URL and open the Student View
- Shareable URL so multiple students can work on same scenario
- The student sees a clean interface where the hidden truth is completely concealed.
  
### Type "Assess surroundings for safety"
- Different action agent used to assess the students action in free text
- Student types an action, aligned to wfr curriculum which reveals hidden information (rattlesnake)

### Switch to mlflow UI with gepa optimzation
- SMEs warned me the action agent was too generous and simulations ended too quickly
- To fix this, I used 4 LLM judges and GEPA optimization in MLflow to align the agent's prompt to expert feedback

### Switch to a tab with a pre-completed scenario
- Final debrief agent provides actionable feedback to student to prep them for the exam

### Conclusion
- Tech stack is chainlit UI, langgraph orchestration, pydantic AI, mlflow tracing/optimization and openrouter for inference
- Summit sim is multi-agent Ai system with HITL to build infinitely many WFR scenarios and provide interactive gameplay for students to prep for WFR exams
