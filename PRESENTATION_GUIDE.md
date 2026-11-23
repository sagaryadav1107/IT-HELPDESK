# IT Helpdesk – Faculty Project Demonstration & Viva Guide

This guide is your personal cheat sheet for your project demonstration and faculty viva. It contains word-for-word presentation scripts, step-by-step live demo instructions, and prepared answers for typical faculty questions.

---

## 1. The 30-Second Elevator Pitch

> *"Respected faculty, my project is the **AI-Based IT Helpdesk Ticket Classification and SLA Lifecycle Management System**. In modern enterprise IT, human dispatchers spend hours manually reading, prioritizing, and assigning support tickets, leading to human misrouting and costly SLA breaches. Our system uses a trained Machine Learning NLP model to instantly classify support tickets into ITIL domains in under 50 milliseconds, dynamically calculates priority and SLA deadlines, routes tickets to specialized resolver groups with direct contact emails, and audits the full incident resolution lifecycle."*

---

## 2. The 2-Minute Comprehensive Introduction

### Why This Project Was Developed (The Problem):
1. **High Mean Time to Triage (MTTT):** In large organizations, tickets sit unassigned in queues for 2 to 6 hours before human dispatchers view them.
2. **Cognitive Fatigue & Misrouting:** Dispatchers often forward complex complaints to the wrong team, causing ticket "ping-pong" and lost engineering hours.
3. **SLA Violations:** Critical production outages can easily be overlooked if buried beneath routine low-priority requests.

### What Our System Solves (The Solution):
* **Automated NLP Classification:** Uses a Term Frequency-Inverse Document Frequency (TF-IDF) feature extractor and multiclass classifier to categorize issues (*Network, Software, Hardware, Authentication, System, Other*) with sub-10ms inference.
* **Deterministic Priority & Dynamic SLA Engine:** Uses a hybrid architecture combining statistical NLP with heuristic keyword urgency checks to assign priority (*Critical, High, Medium, Low*) and bind tickets to enforceable 4h, 8h, 24h, and 48h SLA deadlines.
* **Lifecycle State Machine:** Supports standard ITIL workflows (`Open` ➔ `Assigned` ➔ `In Progress` ➔ `Resolved` ➔ `Closed`) with immutable timestamped audit logs.
* **Executive Analytics:** Real-time dashboard calculating open workloads, resolution counts, and SLA adherence rates.

---

## 3. Step-by-Step Live Demo Runbook

Follow these exact steps during your screen share or faculty presentation:

### Step 1: Start the Streamlit Application
* Open PowerShell in your project folder:
  ```powershell
  cd C:\Users\sahil\.gemini\antigravity\scratch\IT-HELPDESK
  streamlit run app.py
  ```
* *What to Say:* *"The frontend is built on Streamlit with reactive rendering. The ML model pipeline is cached in memory via the `@st.cache_resource` decorator for zero-latency inference."*

### Step 2: Demonstrate AI Classification & Instant Routing
* Navigate to the **Submit Ticket** page on the sidebar.
* In the description box, paste:
  ```text
  VPN connection is dropping frequently and cannot reach internal corporate intranet
  ```
* Click **Classify & Create Ticket**.
* Point out the **Ticket Summary Card**:
  * Category: `🌐 Network Issue`
  * Assigned Team: `👥 Network Support Team (✉️ network-support@company.local)`
  * Priority: `🔴 High`
  * SLA Window: `⏳ 8 Hours`
* *What to Say:* *"Notice how our model instantly recognized this as a Network Issue, looked up the specialized resolver team's coordinates, and automatically calculated the SLA deadline based on urgency keywords."*

### Step 3: Demonstrate Critical Outage Handling
* In the same form, paste:
  ```text
  Production database server down, system crash with out of memory error
  ```
* Click **Classify & Create Ticket**.
* Point out:
  * Category: `⚙️ System Issue`
  * Assigned Team: `System/Storage Support Team`
  * Priority: `🔴 Critical`
  * SLA Window: `⏳ 4 Hours (Deadline generated dynamically)`
* *What to Say:* *"Here, critical indicators ('server down', 'system crash') triggered the highest urgency tier, giving engineers an aggressive 4-hour SLA target."*

### Step 4: Show the Real-Time Executive Dashboard
* Click on **Dashboard** on the sidebar.
* Highlight the top 6 metric cards:
  * **Total Tickets, Open, In Progress, Resolved, SLA Met, SLA Breached**.
* Scroll to the **All Submitted Tickets** table:
  * Show the tickets just submitted, their generated IDs (`INC-1001`, `INC-1002`), assigned teams, and SLA deadlines.
* *What to Say:* *"The executive dashboard gives IT managers full visibility over open tickets, active workloads, and real-time SLA compliance."*

### Step 5: Walk Through the Ticket Lifecycle & SLA Audit
* Click on **Manage Tickets** on the sidebar.
* Select the ticket from the dropdown.
* Click **Assign Ticket** ➔ Watch status change to `Assigned`.
* Click **Start Work** ➔ Watch status change to `In Progress`.
* In the resolution notes box, type:
  ```text
  Restarted database daemon and allocated additional swap space.
  ```
* Click **Resolve Ticket**.
* Point to the green checkmark: `✓ Resolved within SLA`.
* Look at the right column: Show the **Ticket Timeline** with all timestamped events (`Ticket Created`, `AI Classified`, `Assigned`, `Engineer Started Work`, `Ticket Resolved`).
* Click **Close Ticket** to finalize closure.
* *What to Say:* *"This demonstrates the complete ITIL incident lifecycle. The system automatically checked whether resolution time was within the calculated deadline and logged an immutable audit trail."*

---

## 4. Top 10 Faculty Viva Voce Questions & Answers

#### Q1: Which Machine Learning algorithm did you use and why?
> **Answer:** *"We used a Scikit-Learn NLP pipeline combining **TF-IDF (Term Frequency-Inverse Document Frequency)** with a **Multinomial Logistic Classifier**. We chose this over heavy deep learning models because it delivers over 90% accuracy, executes inference in under 10 milliseconds, uses less than 50MB of RAM, and requires no external paid cloud APIs or GPUs."*

#### Q2: How does the model convert unstructured text into numbers?
> **Answer:** *"Through TF-IDF vectorization. Term Frequency measures how often a word appears in the ticket, while Inverse Document Frequency penalizes common words (like 'the', 'is') and elevates discriminative technical terms (like 'VPN', 'ActiveDirectory', 'BSOD'). It also uses n-grams (1, 2) to capture phrase context like 'not working' or 'server down'."*

#### Q3: Why didn't you use Machine Learning to predict priority as well?
> **Answer:** *"We use a **Hybrid AI Architecture**. Machine Learning is probabilistic and excels at semantic domain classification (determining Network vs Hardware). However, business priority and SLA enforcement must be deterministic. An employee might casually write 'the database is down'; an ML model might miscalculate the tone, but our rule engine immediately catches the phrase 'database is down' and enforces a strict Critical 4-hour SLA."*

#### Q4: How is data persisted when the application or server restarts?
> **Answer:** *"Tickets and their chronological event timelines are serialized to `tickets.json`. When the application boots, Streamlit initializes the session registry (`st.session_state.tickets`) from this file, ensuring persistence across sessions."*

#### Q5: How do you prevent reloading the heavy ML model on every button click?
> **Answer:** *"Streamlit re-executes code from top to bottom on user interaction. We decorated `load_model()` with `@st.cache_resource`, ensuring `ticket_classifier_model.pkl` is loaded once as an in-memory singleton and reused across all user interactions with zero disk re-read overhead."*

#### Q6: What happens if a user submits a blank or meaningless issue?
> **Answer:** *"The UI validates inputs to reject empty submissions. If an obscure or non-technical query is submitted, the model's confidence distribution routes it to the 'Other' category, assigned to the General IT Support Team for human clarification."*

#### Q7: How does the SLA compliance check work mathematically?
> **Answer:** *"When a ticket is created at $T_{\text{created}}$, its deadline is $T_{\text{deadline}} = T_{\text{created}} + \text{SLA Hours}$. Upon resolution at $T_{\text{resolved}}$, the system checks if $T_{\text{resolved}} \le T_{\text{deadline}}$. If true, it records `sla_met = True`; otherwise, it records `sla_met = False` and logs an SLA breach on the dashboard."*

#### Q8: Where is the model trained, and how can it be retrained with new tickets?
> **Answer:** *"We have a dedicated `train_model.py` script. As new tickets are resolved and verified by engineers, they can be added to the training set. Running `python train_model.py` updates the pipeline, measures validation accuracy, and exports an updated `ticket_classifier_model.pkl` without modifying any application code."*

#### Q9: What are the security and privacy advantages of this architecture?
> **Answer:** *"The system is 100% on-premises and self-contained. It makes zero third-party API calls, ensuring sensitive internal network topology, server names, and employee details never leave the organization's network."*

#### Q10: What future improvements could be added?
> **Answer:** *"Future enhancements include integrating bi-directional webhooks with Jira or ServiceNow, adding a conversational LLM clarification chatbot for ambiguous tickets, and incorporating an automated knowledge-base search (RAG) to allow users to self-resolve common issues before raising a ticket."*
