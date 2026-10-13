import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
from groq import Groq

# Page Configuration
st.set_page_config(
    page_title="Shamareekh Safety - Smart Construction Training",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header { font-size: 26px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-header { font-size: 15px; color: #475569; margin-bottom: 20px; }
    .card { background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 15px; }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize Session State
if "observations" not in st.session_state:
  st.session_state.observations = [{
      "id": "OBS-1001",
      "date": "2026-10-12",
      "location": "Sector B - Tower 3, Floor 8",
      "hazard_type": "Working at Heights",
      "raw_notes": (
          "Workers on edge scaffolding without safety harness secured."
      ),
      "risk_level": "High",
      "preventive_measures": (
          "Enforce 100% tie-off policy; install edge guardrails; inspect harness"
          " anchorage points."
      ),
      "status": "Approved",
      "assigned_role": "Scaffolder / General Worker",
      "worker_language": "Arabic",
      "training_completed": True,
      "assessment_score": 100,
      "resolved": True,
  }]

# Sidebar - Settings & Groq API
st.sidebar.image(
    "https://www.ssco.edu.sa/ar/about-us",
    caption="Shamareekh Educational Platform",
)
st.sidebar.title("Configuration")

# Fetch API Key from Streamlit Secrets or Sidebar Input
default_key = st.secrets.get("GROQ_API_KEY", "") if "GROQ_API_KEY" in st.secrets else ""
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    value=default_key,
    type="password",
    help="Enter your free Groq API key from console.groq.com",
)

if not groq_api_key:
  st.sidebar.warning("⚠️ Please enter your Groq API Key to enable AI agents.")

client = Groq(api_key=groq_api_key) if groq_api_key else None

# Safe API Completion Helper with Model Fallbacks
PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-120b"


def call_groq_json(client_obj, prompt, temperature=0.2):
  """Calls Groq API safely with fallback models if a NotFoundError occurs."""
  models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL, "gemma2-9b-it"]

  for model_id in models_to_try:
    try:
      response = client_obj.chat.completions.create(
          model=model_id,
          messages=[{"role": "user", "content": prompt}],
          response_format={"type": "json_object"},
          temperature=temperature,
      )
      return json.loads(response.choices[0].message.content)
    except Exception as e:
      if "404" in str(e) or "NotFoundError" in type(e).__name__:
        continue  # Try the next model
      else:
        raise e

  raise RuntimeError(
      "All available Groq model endpoints failed. Please check your API key"
      " and status at console.groq.com"
  )


# Helper AI Agent Functions
def run_hazard_agent(notes, location):
  prompt = f"""
    You are an expert Construction Safety Inspector Agent for the Shamareekh Educational Platform.
    Analyze the following site observation notes:
    Location: {location}
    Notes: {notes}

    Respond strictly with a valid JSON object containing:
    {{
        "hazard_type": "Type of hazard identified",
        "risk_level": "High/Medium/Low",
        "preventive_measures": "3 clear action steps to rectify the hazard"
    }}
    """
  return call_groq_json(client, prompt, temperature=0.2)


def run_training_agent(hazard, role, language):
  prompt = f"""
    You are a Safety Training Curriculum Agent for Shamareekh Educational Platform.
    Create a micro-learning safety module based on:
    Hazard: {hazard}
    Worker Role: {role}
    Target Language: {language}

    Respond strictly with a valid JSON object:
    {{
        "title": "Module Title",
        "key_safety_rules": ["Rule 1", "Rule 2", "Rule 3"],
        "scenario_question": "A practical scenario question testing comprehension",
        "correct_answer": "Brief explanation of correct practical action"
    }}
    """
  return call_groq_json(client, prompt, temperature=0.3)


def run_assessment_agent(user_answer, correct_answer):
  prompt = f"""
    You are a Safety Assessment Evaluator Agent.
    Evaluate the worker's practical answer against the required safety action.
    Target Correct Action: {correct_answer}
    Worker Answer: {user_answer}

    Respond strictly with a valid JSON object:
    {{
        "score": 100 or 0,
        "passed": true/false,
        "feedback": "Constructive feedback in plain language"
    }}
    """
  return call_groq_json(client, prompt, temperature=0.1)


# Main Interface
st.markdown(
    '<div class="main-header">Shamareekh Safety: From Site Hazards to Smart'
    " Training</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Integrated AI Module for Occupational Safety,'
    " Certificates & Site Hazard Tracking</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Executive Dashboard",
    "📝 1. Field Observation",
    "🔍 2. Safety Review",
    "🎓 3. Worker Training",
    "📜 4. Certificate Issuance",
])

# TAB 1: EXECUTIVE DASHBOARD
with tabs[0]:
  st.subheader("Construction Site Safety & Training Analytics")
  df = pd.DataFrame(st.session_state.observations)

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Total Observations", len(df))
  col2.metric("Approved Hazards", len(df[df["status"] == "Approved"]))
  col3.metric("Training Completed", len(df[df["training_completed"] == True]))
  col4.metric("Resolved Hazards", len(df[df["resolved"] == True]))

  st.divider()

  col_chart1, col_chart2 = st.columns(2)
  with col_chart1:
    if not df.empty:
      fig_risk = px.pie(
          df,
          names="risk_level",
          title="Hazard Risk Distribution",
          color_discrete_sequence=px.colors.sequential.RdBu,
      )
      st.plotly_chart(fig_risk, use_container_width=True)
  with col_chart2:
    if not df.empty:
      fig_status = px.bar(
          df,
          x="hazard_type",
          color="status",
          title="Hazard Status by Category",
          barmode="group",
      )
      st.plotly_chart(fig_status, use_container_width=True)

# TAB 2: FIELD OBSERVATION
with tabs[1]:
  st.subheader("Record Site Safety Observation")
  st.write(
      "Safety officers upload notes/photos from construction sites for"
      " automated AI analysis."
  )

  with st.form("observation_form"):
    location = st.text_input(
        "Site Location / Zone",
        placeholder="e.g., Tower A - Floor 12 West Wing",
    )
    raw_notes = st.text_area(
        "Field Notes / Description",
        placeholder=(
            "e.g., Worker operating near unprotected edge without secondary"
            " safety harness lanyard..."
        ),
    )
    uploaded_file = st.file_uploader(
        "Attach Site Photo (Optional)", type=["jpg", "jpeg", "png"]
    )
    submitted = st.form_submit_button("Analyze Hazard with AI Agent")

    if submitted:
      if not client:
        st.error("Please enter a valid Groq API Key in the sidebar first!")
      elif raw_notes and location:
        with st.spinner("Agent 1 (Safety Inspection Agent) analyzing hazard..."):
          try:
            ai_result = run_hazard_agent(raw_notes, location)
            new_obs = {
                "id": f"OBS-{1001 + len(st.session_state.observations)}",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "location": location,
                "hazard_type": ai_result.get("hazard_type", "General Hazard"),
                "raw_notes": raw_notes,
                "risk_level": ai_result.get("risk_level", "Medium"),
                "preventive_measures": ai_result.get(
                    "preventive_measures", ""
                ),
                "status": "Pending Specialist Review",
                "assigned_role": "General Construction Worker",
                "worker_language": "English",
                "training_completed": False,
                "assessment_score": 0,
                "resolved": False,
            }
            st.session_state.observations.append(new_obs)
            st.success("Hazard observation recorded and analyzed successfully!")
          except Exception as e:
            st.error(f"Error calling AI agent: {str(e)}")
      else:
        st.error("Please fill in both site location and field notes.")

# TAB 3: SAFETY SPECIALIST REVIEW
with tabs[2]:
  st.subheader("Safety Specialist Approval Pipeline")
  pending = [
      obs
      for obs in st.session_state.observations
      if obs["status"] == "Pending Specialist Review"
  ]

  if not pending:
    st.info("No observations pending review.")
  else:
    for idx, obs in enumerate(pending):
      with st.expandable(
          f"{obs['id']} - {obs['hazard_type']} ({obs['location']})",
          expanded=True,
      ):
        st.write(f"**Notes:** {obs['raw_notes']}")
        st.write(f"**AI Suggested Risk Level:** {obs['risk_level']}")

        edited_measures = st.text_area(
            "Review/Edit Preventive Measures",
            value=obs["preventive_measures"],
            key=f"meas_{obs['id']}",
        )
        col_r, col_l = st.columns(2)
        assigned_role = col_r.selectbox(
            "Target Worker Role",
            [
                "Scaffolder / General Worker",
                "Crane Operator",
                "Electrician",
                "Welder",
            ],
            key=f"role_{obs['id']}",
        )
        worker_lang = col_l.selectbox(
            "Worker Preferred Language",
            ["English", "Arabic", "Urdu", "Hindi"],
            key=f"lang_{obs['id']}",
        )

        if st.button("Approve & Link to Training", key=f"btn_{obs['id']}"):
          obs["preventive_measures"] = edited_measures
          obs["assigned_role"] = assigned_role
          obs["worker_language"] = worker_lang
          obs["status"] = "Approved"
          st.success(
              f"Observation {obs['id']} approved and assigned for training."
          )
          st.rerun()

# TAB 4: WORKER TRAINING & ASSESSMENT
with tabs[3]:
  st.subheader("Role & Language Tailored Training Module")
  approved_obs = [
      obs
      for obs in st.session_state.observations
      if obs["status"] == "Approved" and not obs["training_completed"]
  ]

  if not approved_obs:
    st.info("No active training sessions required.")
  else:
    selected_id = st.selectbox(
        "Select Hazard Case to Begin Training", [o["id"] for o in approved_obs]
    )
    current_obs = next(o for o in approved_obs if o["id"] == selected_id)

    st.markdown(
        f"**Target Role:** {current_obs['assigned_role']} | **Language:**"
        f" {current_obs['worker_language']}"
    )

    if st.button("Generate Tailored Training Content", key="gen_train"):
      if not client:
        st.error("Please enter a valid Groq API Key in the sidebar first!")
      else:
        with st.spinner(
            "Agent 2 (Curriculum Agent) tailoring micro-learning module..."
        ):
          try:
            train_data = run_training_agent(
                current_obs["hazard_type"],
                current_obs["assigned_role"],
                current_obs["worker_language"],
            )
            st.session_state[f"train_data_{selected_id}"] = train_data
          except Exception as e:
            st.error(f"Error generating training: {str(e)}")

    if f"train_data_{selected_id}" in st.session_state:
      tdata = st.session_state[f"train_data_{selected_id}"]
      st.markdown(f"### {tdata['title']}")
      st.markdown("#### Key Safety Guidelines:")
      for rule in tdata.get("key_safety_rules", []):
        st.markdown(f"- {rule}")

      st.divider()
      st.markdown("#### Scenario-Based Assessment")
      st.write(tdata.get("scenario_question", ""))
      worker_ans = st.text_area("Your Response / Practical Action:")

      if st.button("Submit Assessment"):
        if not client:
          st.error("Please enter a valid Groq API Key in the sidebar first!")
        else:
          with st.spinner(
              "Agent 3 (Assessment Agent) evaluating practical scenario"
              " response..."
          ):
            try:
              eval_res = run_assessment_agent(
                  worker_ans, tdata.get("correct_answer", "")
              )
              if eval_res.get("passed"):
                st.balloons()
                st.success(
                    f"Passed! Score: {eval_res.get('score')}% -"
                    f" {eval_res.get('feedback')}"
                )
                current_obs["training_completed"] = True
                current_obs["assessment_score"] = eval_res.get("score", 100)
                current_obs["resolved"] = True
              else:
                st.error(
                    f"Score: {eval_res.get('score')}% -"
                    f" {eval_res.get('feedback')}. Please review and try again."
                )
            except Exception as e:
              st.error(f"Error evaluating assessment: {str(e)}")

# TAB 5: CERTIFICATE ISSUANCE
with tabs[4]:
  st.subheader("Professional Practice Certification")
  completed = [
      obs
      for obs in st.session_state.observations
      if obs["training_completed"]
  ]

  if not completed:
    st.info(
        "No completed training assessments eligible for certificate issuance"
        " yet."
    )
  else:
    for c in completed:
      st.markdown(
          f"""
            <div class="card" style="text-align: center; border: 2px solid #1E3A8A;">
                <h2 style="color: #1E3A8A; margin-bottom: 0;">Shamareekh Educational Platform</h2>
                <p style="color: #64748B;">Occupational Safety Professional Practice Certificate</p>
                <hr>
                <p>This certifies that the worker assigned to <b>{c['assigned_role']}</b> has successfully completed site-specific practical hazard training for:</p>
                <h3>{c['hazard_type']}</h3>
                <p>Verification Score: <b>{c['assessment_score']}%</b> | Date: {c['date']}</p>
                <p style="font-size: 11px; color: #94A3B8;">Certificate ID: CERT-{c['id']} | Shamareekh Safety AI Verified</p>
            </div>
            """,
          unsafe_allow_html=True,
      )
