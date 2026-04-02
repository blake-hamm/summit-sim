### Show the main Chainlit UI
- Wilderness First Responder (WFR) courses rely on pre-canned, static scenarios from textbooks.
- To solve this, I collaborated with the Ogden Avalanche Center President to build Summit-Sim
-  an AI-generated simulator that creates infinitely many wilderness emergencies and dynamic gameplay for students

### Click "Create Scenario," pick parameters, and submit
- The instructor sets the parameters, and the Generator agent creates a WFR-aligned scenario.
- With multimodal AI, it generates a unique image per scenario
- Here are all the other ouptus, leveraging pydantic AI to enforce strict input/output types

### Type "add a rattlesnake in the hidden state. do not reveal it in any other fields."
- Using LangGraph's interupt feature, the Teacher is able to revise the scenario, for example 'add a rattlesnake'
- You'll notice hidden information, which is presented only to the teacher and AI
- It will be dynamically revealed during simulation based on student actions
- You'll notice the rattlesnake inforrmation has now been added and the instructor approves

### Open the shareable URL and open the Student View
- This creates a unique URL so multiple students can work on same scenario in different sessions.
- Here is the simulation view which conceals hidden information and allows students to gameplay the scenario
  
### Type "Assess surroundings for safety"
- This uses the Action agent to assess the students text response to WFR curriculum
- Student types an action, and agent reveals information and dynamically progresses through scenario
- Initally, I received negative feedback on the Action agent and the simulation ux

### Switch to mlflow UI with gepa optimzation
- To fix this, I used 4 LLM judges and GEPA optimization in MLflow to align the agent's prompt to expert feedback

### Switch to a tab with a pre-completed scenario
- Here is the 3rd, Debrief agent which provides actionable feedback to students to prep them for the exam

### Conclusion
- Summit sim is multi-agent Ai system with HITL to build infinitely many WFR scenarios
- It provide interactive gameplay for students to prep for WFR exams
