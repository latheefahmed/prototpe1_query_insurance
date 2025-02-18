# app.py

import streamlit as st
import sqlite3
import datetime
import re
import os
import json
import pandas as pd
import spacy
import torch
from difflib import SequenceMatcher
from transformers import T5ForConditionalGeneration, T5Tokenizer

########################################
# 1. INITIALIZE INSURANCE LOGIC MODULE #
########################################

# (Do not change any of the code below)

# Load spaCy model (try large, then fallback to small)
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        raise ImportError("No spaCy model found. Run: python -m spacy download en_core_web_lg or en_core_web_sm")

# File paths (update these if needed)
OVERALL_CSV = "insurance_plans_overall.csv"
DISEASE_CSV = "insurance_plans_by_disease.csv"
TEMPLATES_CSV = "question_templates.csv"

try:
    overall_df = pd.read_csv(OVERALL_CSV)
    disease_df = pd.read_csv(DISEASE_CSV)
    templates_df = pd.read_csv(TEMPLATES_CSV)
except Exception as e:
    raise FileNotFoundError("One or more CSV files could not be loaded: " + str(e))

# Extract known entities
known_plan_names = overall_df["Plan Name"].dropna().unique().tolist()
known_premium_types = ["basic", "lite", "premier"]

if "Disease" in disease_df.columns:
    known_diseases = disease_df["Disease"].dropna().unique().tolist()
else:
    known_diseases = []

# Mappings for calculations and target columns
deductible_mapping = {"basic": 5, "lite": 10, "premier": 15}
copay_mapping = {"basic": 20, "lite": 10, "premier": 5}

level3_mapping = {
    "coverage": "Maximum Coverage",
    "maximum coverage": "Maximum Coverage",
    "deductible": "Deductible",
    "copay": "Co‑pay Percentage",
    "out-of-pocket": "Out-of-Pocket",
    "waiting period": "Waiting Period",
    "claims": "Claims Settled",
    "renewability": "Renewability",
    "hospital": "Hospital Coverage",
    "benefits": "Benefits",
    "benifits": "Benefits",
    "tax": "Tax Redemption"
}
level12_mapping = {
    "monthly payment": "Monthly Payment",
    "deductible": "Deductible",
    "copay": "Co‑pay Percentage",
    "plan term": "Plan Term",
    "tax": "Tax Redemption",
    "benefits": "Benefits",
    "benifits": "Benefits"
}

calc_keywords = {"claimable", "calculate", "calculation", "out-of-pocket", "deductible", "copay", "claim"}

# Load the fine-tuned transformer model
model_path = "./t5_fine_tuned_final"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# ---------------------------
# Utility functions (do not change)
# ---------------------------
def match_keyword_in_prompt(keyword: str, prompt_lower: str, threshold: float = 0.8) -> bool:
    if keyword in prompt_lower:
        return True
    for word in prompt_lower.split():
        if SequenceMatcher(None, keyword, word).ratio() >= threshold:
            return True
    return False

def fuzzy_extract_plan_name(prompt: str, known_names: list, threshold: float = 0.7) -> str:
    prompt_lower = prompt.lower()
    for name in known_names:
        if name.lower() in prompt_lower:
            return name
    for word in prompt_lower.split():
        for name in known_names:
            if SequenceMatcher(None, name.lower(), word).ratio() >= threshold:
                return name
    return None

def fuzzy_extract_disease(prompt: str, known_diseases: list, threshold: float = 0.7) -> str:
    prompt_lower = prompt.lower()
    for word in prompt_lower.split():
        for dis in known_diseases:
            if SequenceMatcher(None, dis.lower(), word).ratio() >= threshold:
                return dis
    return None

def extract_entities(prompt: str) -> dict:
    entities = {
        "plan_name": None,
        "premium_type": None,
        "disease": None,
        "amount": None,
        "target_keywords": set()
    }
    prompt_lower = prompt.lower()
    doc = nlp(prompt)

    # Extract monetary amounts using spaCy NER
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            clean_amount = re.sub(r'[^\d.]', '', ent.text)
            try:
                if clean_amount:
                    entities["amount"] = float(clean_amount)
                    break
            except ValueError:
                continue

    # Fallback: regex to find numbers with at least 4 digits.
    if entities["amount"] is None:
        numbers = re.findall(r'\d+', prompt)
        for num in numbers:
            if len(num) >= 4:
                try:
                    entities["amount"] = float(num)
                    break
                except ValueError:
                    continue

    # Direct matching for plan name.
    for name in known_plan_names:
        if name.lower() in prompt_lower:
            entities["plan_name"] = name
            break
    if not entities["plan_name"]:
        fuzzy_name = fuzzy_extract_plan_name(prompt, known_plan_names, threshold=0.7)
        if fuzzy_name:
            entities["plan_name"] = fuzzy_name

    # Premium type extraction.
    for ptype in known_premium_types:
        if ptype in prompt_lower:
            entities["premium_type"] = ptype
            break

    # Disease extraction.
    for dis in known_diseases:
        if dis.lower() in prompt_lower:
            entities["disease"] = dis
            break
    if not entities["disease"]:
        fuzzy_disease = fuzzy_extract_disease(prompt, known_diseases, threshold=0.7)
        if fuzzy_disease:
            entities["disease"] = fuzzy_disease

    # Target keywords extraction.
    for keyword in list(level3_mapping.keys()) + list(level12_mapping.keys()):
        if match_keyword_in_prompt(keyword, prompt_lower):
            if keyword in level3_mapping:
                entities["target_keywords"].add(level3_mapping[keyword])
            if keyword in level12_mapping:
                entities["target_keywords"].add(level12_mapping[keyword])
    if any(match_keyword_in_prompt(kw, prompt_lower) for kw in calc_keywords):
        entities["target_keywords"].add("calculation")
    return entities

def match_template_details(prompt: str, level: str, threshold: float = 0.6) -> dict:
    matched_columns = set()
    matched_types = set()
    prompt_lower = prompt.lower()
    required_cols = ['template', 'level', 'target_column']
    if not all(col in templates_df.columns for col in required_cols):
        return {"target_columns": matched_columns, "question_types": matched_types}
    for idx, row in templates_df.iterrows():
        template_level = str(row['level']).lower().strip()
        if level == "level_3" and template_level != "level_3":
            continue
        elif level != "level_3" and template_level == "level_3":
            continue
        template_text = str(row['template']).lower()
        ratio = SequenceMatcher(None, prompt_lower, template_text).ratio()
        if ratio >= threshold:
            matched_columns.add(row['target_column'])
            if 'question_type' in templates_df.columns:
                matched_types.add(str(row['question_type']).strip())
    return {"target_columns": matched_columns, "question_types": matched_types}

def identify_target_columns(prompt: str, level: str) -> dict:
    prompt_lower = prompt.lower()
    columns = set()
    mapping = level3_mapping if level == "level_3" else level12_mapping
    for key, col in mapping.items():
        if match_keyword_in_prompt(key, prompt_lower):
            columns.add(col)
    template_details = match_template_details(prompt, level)
    columns = columns.union(template_details["target_columns"])
    return {"columns": list(columns), "question_types": list(template_details["question_types"])}

def get_deductible_percent(premium_type: str) -> float:
    return deductible_mapping.get(premium_type.lower(), 0)

def get_copay_percent(premium_type: str) -> float:
    return copay_mapping.get(premium_type.lower(), 0)

# --- Updated process_prompt ---
def process_prompt(prompt: str, default_plan_name=None, default_premium_type=None) -> dict:
    is_if_query = prompt.strip().lower().startswith("if")
    entities = extract_entities(prompt)
    # If no plan name is extracted and a default is provided, use it.
    if entities.get("plan_name") is None and default_plan_name is not None:
        entities["plan_name"] = default_plan_name
        if entities.get("premium_type") is None and default_premium_type is not None:
            entities["premium_type"] = default_premium_type

    if not entities.get("plan_name"):
        return {"error": "Plan name not found in the query."}
    
    if entities.get("disease"):
        df = disease_df
        level = "level_3"
    else:
        df = overall_df
        level = "level_1/2"
    
    query_df = df[df["Plan Name"].str.lower() == entities["plan_name"].lower()]
    if entities.get("premium_type"):
        query_df = query_df[query_df["Premium Type"].str.lower() == entities["premium_type"].lower()]
    if level == "level_3" and entities.get("disease"):
        query_df = query_df[query_df["Disease"].str.lower() == entities["disease"].lower()]
    
    if query_df.empty:
        return {"error": "No matching record found for the given query parameters."}
    
    record = query_df.iloc[0].to_dict()
    
    template_info = identify_target_columns(prompt, level)
    target_cols = template_info["columns"]
    question_types = template_info["question_types"]
    
    if entities.get("amount") is not None:
        if level != "level_3":
            return {"error": "Calculation queries require disease-specific (Level 3) information."}
        try:
            max_coverage = float(record["Maximum Coverage"])
        except Exception as e:
            return {"error": "Maximum Coverage value is not numeric: " + str(e)}
        premium = entities.get("premium_type") if entities.get("premium_type") else "basic"
        deductible_percent = get_deductible_percent(premium)
        copay_percent = get_copay_percent(premium)
        
        condition_result = "Yes" if entities["amount"] <= max_coverage else "No"
        if entities["amount"] <= max_coverage:
            calc_deductible = entities["amount"] * deductible_percent / 100
            updated_amount = entities["amount"] - calc_deductible
            calc_copay = updated_amount * copay_percent / 100
            total_out_of_pocket = calc_deductible + calc_copay
            result = {
                "requested_amount": entities["amount"],
                "maximum_coverage": max_coverage,
                "deductible_percent": deductible_percent,
                "copay_percent": copay_percent,
                "calculated_deductible": round(calc_deductible, 2),
                "calculated_copay": round(calc_copay, 2),
                "total_out_of_pocket": round(total_out_of_pocket, 2),
                "claimable_amount": entities["amount"],
                "question_type": list(set(["calculation"]).union(set(question_types)))
            }
        else:
            calc_deductible = max_coverage * deductible_percent / 100
            updated_coverage = max_coverage - calc_deductible
            calc_copay = updated_coverage * copay_percent / 100
            total_out_of_pocket = calc_deductible + calc_copay
            excess = entities["amount"] - max_coverage
            result = {
                "requested_amount": entities["amount"],
                "maximum_coverage": max_coverage,
                "deductible_percent": deductible_percent,
                "copay_percent": copay_percent,
                "calculated_deductible": round(calc_deductible, 2),
                "calculated_copay": round(calc_copay, 2),
                "total_out_of_pocket": round(total_out_of_pocket, 2),
                "excess_not_covered": round(excess, 2),
                "claimable_amount": max_coverage,
                "question_type": list(set(["calculation"]).union(set(question_types)))
            }
        if is_if_query:
            result["if_condition"] = condition_result
        return result
    else:
        if not target_cols:
            record["question_type"] = question_types
            return record
        else:
            output = {col: record[col] for col in target_cols if col in record}
            output["question_type"] = question_types
            return output

def format_retrieval_sentence(prompt: str, retrieval: dict) -> str:
    if "error" in retrieval:
        return f"Error encountered: {retrieval['error']}"
    
    if any(key in retrieval for key in ["requested_amount", "maximum_coverage", "calculated_deductible"]):
        sentence = (f"For your query '{prompt}', our system evaluated an insurance claim of Rs.{retrieval.get('requested_amount', 'N/A')} "
                    f"against a maximum coverage of Rs.{retrieval.get('maximum_coverage', 'N/A')}. The deductible is Rs.{retrieval.get('calculated_deductible', 'N/A')} "
                    f"and the co‑pay is Rs.{retrieval.get('calculated_copay', 'N/A')}, resulting in a total out‑of‑pocket cost of Rs.{retrieval.get('total_out_of_pocket', 'N/A')}.")
        if "excess_not_covered" in retrieval:
            sentence += f" An excess of Rs.{retrieval.get('excess_not_covered')} is not covered."
        if "if_condition" in retrieval:
            sentence += f" The claim condition evaluated to '{retrieval.get('if_condition')}'."
    else:
        details = "; ".join(f"{k}: {v}" for k, v in retrieval.items() if k != "question_type")
        sentence = f"For your query '{prompt}', the retrieved details are: {details}."
    
    if "question_type" in retrieval and retrieval["question_type"]:
        sentence += f" [Detected Question Type(s): {', '.join(retrieval['question_type'])}]"
    return sentence

def generate_sentence_with_transformer(prompt: str) -> str:
    input_text = "summarize: " + prompt
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return f"For your query '{prompt}', {generated}"


#######################################
# 2. USER & QUERY MANAGEMENT VIA CSV  #
#######################################

# ---------- USER MANAGEMENT ----------

# Define the CSV file to store user credentials and active plans.
USERS_CSV = "users.csv"

# Function to load users from CSV or create an empty DataFrame if file doesn't exist.
def load_users():
    if os.path.exists(USERS_CSV):
        return pd.read_csv(USERS_CSV)
    else:
        df = pd.DataFrame(columns=["username", "password", "active_plans"])
        df.to_csv(USERS_CSV, index=False)
        return df

# Function to add a new user.
def add_user(username, password, active_plans):
    df = load_users()
    # Check if username already exists
    if username in df["username"].values:
        return False
    # Store active_plans as JSON string
    active_plans_json = json.dumps(active_plans)
    new_row = pd.DataFrame([[username, password, active_plans_json]], columns=["username", "password", "active_plans"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(USERS_CSV, index=False)
    return True

# Function to verify user credentials.
def verify_user(username, password):
    df = load_users()
    if username in df["username"].values:
        user_row = df[df["username"] == username]
        if user_row.iloc[0]["password"] == password:
            return True
    return False

# ---------- QUERY HISTORY MANAGEMENT ----------

# Define the CSV file to store query history.
QUERY_HISTORY_CSV = "query_history.csv"

# Function to store a query to the CSV file.
def store_query_csv(username, query, plan_name, premium_type):
    timestamp = datetime.datetime.now().isoformat()
    # If file doesn't exist, create it with appropriate columns.
    if not os.path.exists(QUERY_HISTORY_CSV):
        df = pd.DataFrame(columns=["username", "query", "plan_name", "premium_type", "timestamp"])
        df.to_csv(QUERY_HISTORY_CSV, index=False)
    # Append the new row.
    df = pd.read_csv(QUERY_HISTORY_CSV)
    new_row = pd.DataFrame([[username, query, plan_name, premium_type, timestamp]], columns=["username", "query", "plan_name", "premium_type", "timestamp"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(QUERY_HISTORY_CSV, index=False)

# Function to load query history for a given user.
def load_query_history(username):
    if os.path.exists(QUERY_HISTORY_CSV):
        df = pd.read_csv(QUERY_HISTORY_CSV)
        return df[df["username"] == username]
    else:
        return pd.DataFrame(columns=["username", "query", "plan_name", "premium_type", "timestamp"])


#######################################
# 3. STREAMLIT FRONTEND APPLICATION   #
#######################################

# Set page configuration
st.set_page_config(page_title="Insurance Chat Assistant", page_icon="💬", layout="wide")

# Initialize session state for authentication.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

# ------------------
# Sidebar: User Panel with Login/Sign Up
# ------------------

with st.sidebar:
    st.title("User Panel")
    
    # Create two tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if verify_user(login_username, login_password):
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success("Logged in successfully!")
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("Choose a Username", key="signup_username")
        signup_password = st.text_input("Choose a Password", type="password", key="signup_password")
        
        # Insurance plans provided
        available_plans = [
            "Individual Health Insurance", "Family Health Insurance", "Parents Health Insurance",
            "Protability Insurance", "Children Health Insurance", "Accidental Coverage", "Covid Coverage"
        ]
        st.write("Select your active plans and choose a premium type for each:")
        # Let user select multiple active plans via multiselect.
        selected_plans = st.multiselect("Select Active Plans", available_plans, key="active_plans")
        
        active_plans = {}
        # For each selected plan, let the user choose a premium type.
        for plan in selected_plans:
            premium = st.selectbox(f"Premium Type for {plan}", options=["Basic", "Lite", "Premier"], key=f"premium_{plan}")
            active_plans[plan] = premium
        
        if st.button("Sign Up"):
            if signup_username and signup_password:
                if add_user(signup_username, signup_password, active_plans):
                    st.success("Sign Up successful! You can now log in.")
                else:
                    st.error("Username already exists. Please choose another.")
            else:
                st.error("Please enter both username and password.")
    
    # If logged in, show a dropdown for active plans from the user record.
    if st.session_state.logged_in:
        df_users = load_users()
        user_row = df_users[df_users["username"] == st.session_state.username]
        if not user_row.empty:
            # Load the active plans (stored as JSON) for the logged in user.
            active_plans_json = user_row.iloc[0]["active_plans"]
            try:
                user_active_plans = json.loads(active_plans_json)
            except:
                user_active_plans = {}
            if user_active_plans:
                st.markdown("### Your Active Plans")
                # Build a list of options in the format "Plan Name (Premium)"
                plan_options = [f"{plan} ({ptype})" for plan, ptype in user_active_plans.items()]
                # Store the selected active plan in session state
                selected_plan = st.selectbox("Select an Active Plan", options=plan_options, key="selected_active_plan")
    
    # Suggested queries in the sidebar (for all users)
    st.markdown("### Suggested Queries")
    st.write("- What is the claimable amount for Individual Health Insurance?")
    st.write("- Tell me about the waiting period for Family Health Insurance.")
    st.write("- What benefits are provided under Covid Coverage?")
    st.write("- How much is the deductible for Parents Health Insurance?")
    st.write("- What is the co‑pay percentage for Accidental Coverage?")

# ------------------
# Main Chat Interface
# ------------------
st.title("💬 Query Addresser")
st.markdown(
    """
    **Key Queries:**  
    - Claimable amount  
    - Waiting period  
    - Benifits  
    - Deductibles  
    - Co‑pay percentages  
    ---
    This app lets you interact with our insurance query system.
    
    - **Logged‑in users:** Your query details will be stored and you can view your active plans.
    - **Guests:** You can still query the system, but details will not be stored.
    """
)

# Chat input and output form.
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input("Enter your query here:")
    submitted = st.form_submit_button("Submit Query")

if submitted and user_query:
    with st.spinner("Processing your query..."):
        # If the user is logged in and has selected an active plan,
        # use it as the default if no plan name is provided in the query.
        if st.session_state.logged_in and "selected_active_plan" in st.session_state:
            selected = st.session_state.selected_active_plan
            # Expecting format "Plan Name (Premium)"
            if "(" in selected:
                default_plan_name = selected.split("(")[0].strip()
                default_premium_type = selected.split("(")[1].replace(")", "").strip().lower()
            else:
                default_plan_name = None
                default_premium_type = None
            retrieval_result = process_prompt(user_query, default_plan_name, default_premium_type)
        else:
            retrieval_result = process_prompt(user_query)
        response_sentence = format_retrieval_sentence(user_query, retrieval_result)
        st.markdown("### Response:")
        st.write(response_sentence)
        
        # If the user is logged in, store the query details in the CSV.
        if st.session_state.logged_in:
            plan_name = retrieval_result.get("Plan Name") or retrieval_result.get("plan_name", "")
            premium_type = retrieval_result.get("Premium Type") or retrieval_result.get("premium_type", "")
            store_query_csv(st.session_state.username, user_query, plan_name, premium_type)
            st.info("Your query details have been stored.")

# ------------------
# Display Query History for Logged-in Users
# ------------------
if st.session_state.logged_in:
    st.sidebar.markdown("### Your Query History")
    history_df = load_query_history(st.session_state.username)
    if not history_df.empty:
        # Show up to the 5 most recent queries in a scrollable area.
        history_df = history_df.sort_values(by="timestamp", ascending=False).head(5)
        for idx, row in history_df.iterrows():
            st.sidebar.write(f"**Query:** {row['query']}")
            st.sidebar.write(f"Plan: {row['plan_name']} | Premium: {row['premium_type']} | At: {row['timestamp']}")
            st.sidebar.markdown("---")
    else:
        st.sidebar.write("No history available yet.")
