# 🏔️ Summit-Sim

**Summit-Sim uses human-in-the-loop review to generate curriculum-informed, interactive backcountry emergencies for dynamic Wilderness First Responder (WFR) training.**

![summit-sim](./public/favicon.png)

*Built for the [Weber State AI Hackathon](https://hackathon.weber.edu/) 🐾*

---

## 💡 Novelty

Traditional WFR training relies on static paper scenarios. **Summit-Sim reimagines this with AI**—generating infinite, curriculum-informed emergencies that evolve in real-time based on student decisions.

**What's novel:**

- **Dynamic state machine:** Natural language decision-making with realistic patient condition evolution
- **Multimodal AI:** Each scenario features a unique AI-generated atmospheric image
- **Human-in-the-loop (HITL):** Instructors review and approve scenarios before students engage
- **Cumulative PAS scoring:** Tracks progress across 5 WFR milestones (0-100% completion)
- **Progressive revelation:** Hidden information revealed only through proper assessment
- **GEPA optimization:** MLflow-based prompt optimization using expert judges for continuous improvement

**Real impact:** Provides a dynamic, engaging interface for WFR exam preparation with infinite scenario variety.


## ⚙️ Technicality

A production-ready AI stack designed for medical safety, observability, and latency:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangGraph | Complex state flows, HITL interrupts, checkpointing |
| **Agent Framework** | PydanticAI | Strict medical safety via enforced Pydantic schemas |
| **LLM Provider** | OpenRouter | Multi-model access (Gemini Flash for text, Flash Image for visuals) |
| **Observability** | MLflow | Span tracing, feedback tracking, prompt versioning, GEPA optimization |
| **UI Framework** | Chainlit | Async, reactive Python UI for conversational flows |
| **State Storage** | Redis / DragonflyDB | LangGraph checkpoint persistence (Redis for local, DragonflyDB for production) |
| **Image Generation** | Gemini Flash Image | 16:9 atmospheric scenario visuals |

**Current Architecture:** Four specialized agents (Generator, Image Generator, Action Responder, Debrief) orchestrated via LangGraph with HITL interrupts and MLflow tracing.

**Validation System:** Four MLflow judges (Structure, Scoring, Medical, Continuity) implemented for offline GEPA optimization. Automated runtime validation disabled pending MLflow bug fix #20782.


## 👥 UX

Two distinct, intuitive flows:

### 🎓 For Instructors
Create scenarios by specifying environment, group size, difficulty, and complexity. The AI generates a complete scenario with atmospheric image. Review and approve via HITL workflow—provide feedback to regenerate (up to 3 attempts). Share the unique URL with students.

### 🚑 For Students
Enter a living emergency via shareable link from your instructor. No multiple choice—use natural language to assess patients, check vitals, and apply treatments. The patient's condition evolves realistically based on your actions. Complete the simulation at 80% PAS milestone completion or continue to max turns. Receive detailed debrief with clinical reasoning assessment and teaching points.

**Self-directed Practice:** Students can also generate their own scenarios for independent study. These auto-generate without instructor review and jump directly into simulation. **Important:** Students never see hidden information (actual diagnosis, learning objectives, or complete medical data)—you must discover all medical details through proper assessment, just like in real-world scenarios.


## 🏗️ Infrastructure

Deployed on a production homelab:
- **Kubernetes:** Talos Linux on Proxmox with Cilium CNI
- **Storage:** Ceph, SeaweedFS (S3), DragonflyDB (redis-compatible), PostgreSQL
- **GitOps:** ArgoCD with common Helm chart patterns
- **Ingress:** Cloudflare tunnel for external access and Traefik for internal access


## 🙏 Acknowledgments

Thanks to **Keenan Grady** at the [Ogden Avalanche Center](https://ogdenavalanche.org/) for helping brainstorm this idea at the intersection of AI and wilderness safety.


## 🔗 Connect

- 💼 **[bhamm-lab.com](https://site.bhamm-lab.com/about/)** – About me
- 📖 **[docs.bhamm-lab.com](https://docs.bhamm-lab.com)** – Infrastructure Documentation
- 🏔️ **[ogdenavalanche.org](https://ogdenavalanche.org)** – Backcountry Safety Education
- 💾 **[github](https://github.com/blake-hamm/summit-sim)** – Source code

---

**Try it out:** Instructors create scenarios, students join with a link. Dynamic WFR training powered by AI.
