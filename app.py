import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
import json

# Path to the trained model
MODEL_PATH = 'ticket_classifier_model.pkl'
TICKETS_FILE = 'tickets.json'

# --- Configuration & Mappings ---
CATEGORY_EMOJIS = {
    "Network Issue": "🌐",
    "Software Issue": "💻",
    "Hardware Issue": "🖥️",
    "Login/Authentication": "🔑",
    "System Issue": "⚙️",
    "Other": "❓"
}

# Map model categories to Support Teams and their Emails
TEAM_ASSIGNMENT = {
    "Network Issue": {"team": "Network Support Team", "email": "network-support@company.local"},
    "Hardware Issue": {"team": "Hardware Support Team", "email": "hardware-support@company.local"},
    "Software Issue": {"team": "Software Support Team", "email": "software-support@company.local"},
    "Login/Authentication": {"team": "Identity & Access Management Team", "email": "iam-support@company.local"},
    "System Issue": {"team": "System/Storage Support Team", "email": "sysadmin@company.local"},
    "Other": {"team": "General IT Support Team", "email": "helpdesk@company.local"}
}

SLA_HOURS = {
    "Critical": 4,
    "High": 8,
    "Medium": 24,
    "Low": 48
}

def determine_priority(description):
    desc_lower = description.lower()
    critical_keywords = ["system down", "cannot access", "server down", "crash", "critical", "urgent", "completely blank"]
    high_keywords = ["account locked", "unable to login", "internet not working", "not working", "failing"]
    
    if any(kw in desc_lower for kw in critical_keywords):
        return "Critical"
    elif any(kw in desc_lower for kw in high_keywords):
        return "High"
    elif len(description) > 30:
        return "Medium"
    else:
        return "Low"

# --- Data Management ---
def load_tickets():
    if os.path.exists(TICKETS_FILE):
        try:
            with open(TICKETS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_tickets(tickets):
    with open(TICKETS_FILE, 'w') as f:
        json.dump(tickets, f, indent=4)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def generate_ticket_id(tickets):
    if not tickets:
        return "INC-1001"
    last_id = tickets[-1]["id"]
    last_num = int(last_id.split("-")[1])
    return f"INC-{last_num + 1}"

# --- Main Application ---
def main():
    st.set_page_config(page_title="IT Helpdesk Management", page_icon="🎫", layout="wide")
    
    # Load model and data
    model = load_model()
    if 'tickets' not in st.session_state:
        st.session_state.tickets = load_tickets()
        
    tickets = st.session_state.tickets

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Submit Ticket", "Dashboard", "Manage Tickets"])
    
    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Please run `train_model.py` first.")
        return

    # ----------------------------------------
    # PAGE: DASHBOARD
    # ----------------------------------------
    if page == "Dashboard":
        st.title("📊 IT Helpdesk Ticket Dashboard")
        
        # Calculate Metrics
        total_tickets = len(tickets)
        open_tickets = sum(1 for t in tickets if t["status"] == "Open")
        in_progress = sum(1 for t in tickets if t["status"] in ["Assigned", "In Progress"])
        resolved_tickets = [t for t in tickets if t["status"] in ["Resolved", "Closed"]]
        resolved_count = len(resolved_tickets)
        
        sla_met = sum(1 for t in resolved_tickets if t.get("sla_met") == True)
        sla_breached = sum(1 for t in resolved_tickets if t.get("sla_met") == False)
        
        # Display Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Tickets", total_tickets)
        col2.metric("Open", open_tickets)
        col3.metric("In Progress", in_progress)
        col4.metric("Resolved", resolved_count)
        col5.metric("SLA Met", sla_met)
        col6.metric("SLA Breached", sla_breached)
        
        st.markdown("---")
        st.subheader("All Submitted Tickets")
        
        if total_tickets > 0:
            df = pd.DataFrame(tickets)
            # Reorder and format columns for display
            display_df = df[["id", "description", "category", "assigned_team", "priority", "status", "sla_deadline", "resolution_time"]]
            display_df.columns = ["Ticket ID", "Issue", "Category", "Assigned Team", "Priority", "Status", "SLA Deadline", "Resolution Time"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No tickets have been submitted yet. Go to 'Submit Ticket' to create one.")

    # ----------------------------------------
    # PAGE: SUBMIT TICKET
    # ----------------------------------------
    elif page == "Submit Ticket":
        st.title("🎫 Submit a New Support Ticket")
        st.markdown("Describe your issue below, and our AI will classify, prioritize, and route it to the appropriate team.")
        
        with st.form("ticket_form"):
            user_input = st.text_area("Describe your technical issue:", 
                                      placeholder="e.g., My account is locked and I cannot access the portal.",
                                      height=150)
            submit_button = st.form_submit_button(label="Classify & Create Ticket")

        if submit_button:
            if user_input.strip() == "":
                st.warning("Please enter a description for your issue.")
            else:
                # 1. AI Classification
                prediction = model.predict([user_input])[0]
                
                # 2. Assignment & Priority Rules
                team_info = TEAM_ASSIGNMENT.get(prediction, {"team": "General IT Support Team", "email": "helpdesk@company.local"})
                assigned_team = team_info["team"]
                assigned_email = team_info["email"]
                priority = determine_priority(user_input)
                
                # 3. Generating Data
                ticket_id = generate_ticket_id(tickets)
                created_time = datetime.now()
                sla_hours = SLA_HOURS[priority]
                sla_deadline = created_time + timedelta(hours=sla_hours)
                
                new_ticket = {
                    "id": ticket_id,
                    "description": user_input,
                    "category": prediction,
                    "assigned_team": assigned_team,
                    "assigned_email": assigned_email,
                    "priority": priority,
                    "status": "Open",
                    "created_time": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sla_deadline": sla_deadline.strftime("%Y-%m-%d %H:%M:%S"),
                    "resolution_time": None,
                    "resolution_notes": None,
                    "sla_met": None,
                    "timeline": [{"event": "Ticket Created", "time": created_time.strftime("%Y-%m-%d %H:%M:%S")},
                                 {"event": "AI Classified", "time": created_time.strftime("%Y-%m-%d %H:%M:%S")}]
                }
                
                tickets.append(new_ticket)
                st.session_state.tickets = tickets
                save_tickets(tickets)
                
                # Display Professional Summary Card
                st.success("Ticket successfully created and routed!")
                
                st.markdown("### 📋 Ticket Summary Card")
                with st.container():
                    st.markdown(f"""
                    **Ticket ID:** `{ticket_id}`  
                    **Issue Description:** {user_input}  
                    **AI Category:** {CATEGORY_EMOJIS.get(prediction, '')} {prediction}  
                    **Assigned Support Team:** 👥 {assigned_team} *( ✉️ {assigned_email} )*  
                    **Priority:** 🔴 {priority}  
                    **SLA Target:** ⏳ {sla_hours} Hours ({sla_deadline.strftime("%Y-%m-%d %H:%M:%S")})  
                    **Current Status:** 🟢 Open  
                    **Created Time:** {created_time.strftime("%Y-%m-%d %H:%M:%S")}  
                    """)

    # ----------------------------------------
    # PAGE: MANAGE TICKETS (LIFECYCLE)
    # ----------------------------------------
    elif page == "Manage Tickets":
        st.title("🛠️ Manage Ticket Lifecycle")
        st.markdown("Simulate the resolution process of a ticket for demonstration purposes.")
        
        if not tickets:
            st.warning("No tickets available to manage.")
            return
            
        ticket_ids = [t["id"] for t in tickets]
        selected_id = st.selectbox("Select a Ticket to Manage:", ticket_ids)
        
        # Find the selected ticket
        ticket = next((t for t in tickets if t["id"] == selected_id), None)
        ticket_index = tickets.index(ticket)
        
        if ticket:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"Ticket: {ticket['id']}")
                st.markdown(f"**Issue:** {ticket['description']}")
                assigned_email_str = ticket.get('assigned_email', 'helpdesk@company.local')
                st.markdown(f"**Category:** {ticket['category']} | **Team:** {ticket['assigned_team']} (✉️ {assigned_email_str})")
                st.markdown(f"**Priority:** {ticket['priority']} | **SLA Deadline:** {ticket['sla_deadline']}")
                st.markdown(f"**Current Status:** `{ticket['status']}`")
                
                if ticket.get('resolution_notes'):
                    st.info(f"**Resolution Notes:** {ticket['resolution_notes']}")
                
                if ticket['sla_met'] is True:
                    st.success("✓ Resolved within SLA")
                elif ticket['sla_met'] is False:
                    st.error("⚠ SLA Breached")
                    
                st.markdown("---")
                st.markdown("### Actions")
                
                # Lifecycle Buttons
                if ticket["status"] == "Open":
                    if st.button("Assign Ticket"):
                        ticket["status"] = "Assigned"
                        ticket["timeline"].append({"event": f"Assigned to {ticket['assigned_team']}", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                        tickets[ticket_index] = ticket
                        save_tickets(tickets)
                        st.rerun()
                        
                elif ticket["status"] == "Assigned":
                    if st.button("Start Work"):
                        ticket["status"] = "In Progress"
                        ticket["timeline"].append({"event": "Engineer Started Work", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                        tickets[ticket_index] = ticket
                        save_tickets(tickets)
                        st.rerun()
                        
                elif ticket["status"] == "In Progress":
                    with st.form("resolve_form"):
                        resolution_notes = st.text_input("Resolution Notes:", placeholder="e.g., Password reset completed and account unlocked.")
                        resolve_btn = st.form_submit_button("Resolve Ticket")
                        
                        if resolve_btn:
                            ticket["status"] = "Resolved"
                            ticket["resolution_notes"] = resolution_notes
                            resolve_time = datetime.now()
                            ticket["resolution_time"] = resolve_time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Calculate SLA
                            sla_deadline = datetime.strptime(ticket["sla_deadline"], "%Y-%m-%d %H:%M:%S")
                            if resolve_time <= sla_deadline:
                                ticket["sla_met"] = True
                            else:
                                ticket["sla_met"] = False
                                
                            ticket["timeline"].append({"event": "Ticket Resolved", "time": resolve_time.strftime("%Y-%m-%d %H:%M:%S")})
                            tickets[ticket_index] = ticket
                            save_tickets(tickets)
                            st.rerun()
                            
                elif ticket["status"] == "Resolved":
                    if st.button("Close Ticket"):
                        ticket["status"] = "Closed"
                        ticket["timeline"].append({"event": "Ticket Closed", "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                        tickets[ticket_index] = ticket
                        save_tickets(tickets)
                        st.rerun()
                elif ticket["status"] == "Closed":
                    st.success("This ticket has been successfully closed.")

            with col2:
                st.subheader("Ticket Timeline")
                for event in ticket["timeline"]:
                    st.markdown(f"↓ **{event['event']}**  \n*( {event['time']} )*")


if __name__ == "__main__":
    main()
