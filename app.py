import streamlit as st
import json
import os
import requests

# Load configuration from config.json file
def load_config():
    config_file_path = os.path.join('conf', 'config.json')
    with open(config_file_path, 'r') as file:
        return json.load(file)

# Load tweaks from file
def load_tweaks():
    tweaks_file_path = os.path.join('tweaks', 'tweaks.json')
    with open(tweaks_file_path, 'r') as file:
        return json.load(file)

# Function to run flow with Streamlit inputs
def run_flow(message: str, tweaks: dict, config: dict, output_type: str = "chat", input_type: str = "chat"):
    BASE_API_URL = "https://api.langflow.astra.datastax.com"
    langflow_id = config['langflow_id']
    flow_id = config['flow_id']
    api_url = f"{BASE_API_URL}/lf/{langflow_id}/api/v1/run/{flow_id}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
        "tweaks": tweaks
    }

    application_token = os.getenv('ASTRA_DB_VECTOR_TOKEN')
    headers = {"Authorization": f"Bearer {application_token}", "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Return the JSON response to be processed
    else:
        return {"error": f"Request failed with status code {response.status_code}"}

# Function to extract message from the response
def extract_message(response: dict):
    try:
        # Navigate through the nested response to find the message text
        message_text = response['outputs'][0]['outputs'][0]['results']['message']['data']['text']
        return message_text
    except KeyError:
        return "Error: Unable to extract message from the response."

# Streamlit app setup
st.title('LangFlow Chat Application')

# Load configuration file
config = load_config()

# User input text box
user_message = st.text_input('Enter your message')

# Load tweaks but do not display them on the page
tweaks = load_tweaks()

# Add configurable parameters for session-level tweaking
st.sidebar.header("Session-Level Configurations")

# Selectable models (dropdown)
model = st.sidebar.selectbox('Select Model', [
    'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-125'],
    index=0
)
tweaks['OpenAIModel-cU5Dl']['model_name'] = model

# Selectable model_name (dropdown)
model_name = st.sidebar.selectbox('Select Embedding Model', [
    'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'],
    index=0
)
tweaks['OpenAIEmbeddings-Rljdq']['model'] = model_name

# Add temperature parameter (slider)
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=tweaks.get('OpenAIModel-cU5Dl', {}).get('temperature', 0.7), step=0.01)
tweaks['OpenAIModel-cU5Dl']['temperature'] = temperature

# When the user submits a message, call the API
if st.button('Send'):
    if user_message:
        st.write(f"Sending message: {user_message}")

        # Call the run_flow function and get the response
        response = run_flow(user_message, tweaks=tweaks, config=config)
        
        # Extract and display only the message part
        extracted_message = extract_message(response)
        st.write("Response: ", extracted_message)
    else:
        st.write("Please enter a message.")
