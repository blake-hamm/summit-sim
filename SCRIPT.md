### Show the main Chainlit UI
- Wilderness First Responder (WFR) courses rely on pre-canned, static scenarios from textbooks.
- To solve this, I collaborated with the Ogden Avalanche Center President to build Summit-Sim
-  an AI-generated simulator that creates infinitely many wilderness emergencies and dynamic gameplay for students

### Click "Create Scenario," pick parameters, and submit
- The aurthor sets the parameters here, and the Generator agent creates a WFR-aligned scenario.
- With multimodal AI, it creates a unique image for the simulation
- Here are all the other ouptus, leveraging pydantic AI to enforce strict input/output types

### Type "add a rattlesnake in the hidden state. do not reveal it in any other fields."
- Using LangGraph's interupt feature, the Teacher is able to revise the scenario, for example 'add a rattlesnake'
- You'll notice hidden information throughout, which is presented only to the teacher and AI agent
- It is revealed during simulation based on the student actions
- Now, the rattlesnake information is added to the hidden state and the teacher approves

### Open the shareable URL and open the Student View
- This creates a unique url so multiple students can work on same scenario in different sessions.
- Here is the simulation view which conceals hidden information and allows students to gameplay the scenario
- This uses the Action agent to assess the students input to WFR curriculum
  
### Type "Assess surroundings for safety"
- Student types an action-the agent reveals information and dynamically progresses through the scenario
- Initally, I received negative feedback on the Action agent saying it was too generous and would pre-maturely complete a session

### Switch to mlflow UI with gepa optimzation
- To fix this, I used 4 LLM judges and genetic pareto (GEPA) optimization in MLflow to align the agent's prompt to expert feedback

### Switch to a tab with a pre-completed scenario
- Once the scenario is complete, a final, Debrief agent summarizes the session and provides actionable feedback to prep students for the exam

### Conclusion
- Summit sim is a multi-agent Ai system with HITL to build infinite WFR-aligned scenarios
- It provide interactive gameplay for students to supplement their textbook learning
