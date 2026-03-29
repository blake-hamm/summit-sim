# 🏔️ Summit-Sim

**Summit-Sim uses human-in-the-loop review to generate curriculum-informed, interactive backcountry emergencies for dynamic Wilderness First Responder (WFR) training.**

![summit-sim](./public/favicon.png)

*Built for the [Weber State AI Hackathon](https://hackathon.weber.edu/) 🐾*

---

## 💡 Novelty

Traditional Wilderness First Responder (WFR) training relies on static paper scenarios. **Summit-Sim reimagines this with AI**—generating infinite, curriculum-informed emergencies that evolve in real-time based on student decisions.

**What's novel:**
- **Dynamic state machine** natural language decision-making and dynamic scenario adaptation
- **Human-in-the-loop (HITL)** workflow: instructors review and approve scenarios before students see them
- **Cumulative scoring** tracks student progress (0-100%) with dynamic completion detection
- **MLflow observability** captures all traces and human feedback for continuous improvement

**Real impact:** Provides a fun and dynamic interface for student preparation for WFR exams.


## ⚙️ Technicality

A production-ready AI stack designed for latency, safety, and observability:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Complex state flows, instructor interrupts, checkpointing |
| **Agent Framework** | PydanticAI | Strict medical safety via enforced Pydantic schemas |
| **Observability** | MLflow | LLM span tracing, human feedback tracking, prompt versioning |
| **UI Framework** | Chainlit | Async, reactive Python UI for conversational flows |
| **State Storage** | Redis/Dragonfly | LangGraph checkpoint persistence |

**Current Architecture:** Three specialized agents (Generator, Action Responder, Debrief) orchestrated via LangGraph with HITL interrupts.

**Planned:** MLflow automatic validation judges for medical accuracy assessment.


## 👥 UX

Two distinct, intuitive flows:

### 🎓 For Teachers
Create scenarios by specifying environment, group size, and difficulty. Review AI-generated emergencies before students see them. Approve or provide feedback to improve future generations.

### 🚑 For Students
Enter a living emergency. No multiple choice—use natural language to assess patients, check vitals, and apply treatments. The patient's condition evolves realistically based on your timeline and interventions.


## 🏗️ Infrastructure

Deployed on a production homelab:
- **Kubernetes:** Talos Linux on Proxmox with Cilium CNI
- **Storage:** Ceph, SeaweedFS (S3), Dragonflydb (redis), Postgresql
- **GitOps:** ArgoCD with common Helm chart patterns
- **Ingress:** Cloudflare tunnel for external access and traefik for internal access


## 🙏 Acknowledgments

Thanks to **Keenan Grady** at the [Ogden Avalanche Center](https://ogdenavalanche.org/) for helping brainstorm this idea at the intersection of AI and wilderness safety.


## 🔗 Connect

- 💼 **[bhamm-lab.com](https://site.bhamm-lab.com/about/)** – About me
- 📖 **[docs.bhamm-lab.com](https://docs.bhamm-lab.com)** – Infrastructure Documentation
- 🏔️ **[ogdenavalanche.org](https://ogdenavalanche.org)** – Backcountry Safety Education

---

**Try it out:** Teachers create scenarios, students join with a link. Dynamic WFR training powered by AI.
