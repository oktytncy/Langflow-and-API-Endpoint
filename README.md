# Langflow and API Endpoint

## Prerequisites
This application assumes you have access to:
1. [DataStax Astra DB](https://astra.datastax.com) (you can sign up through your Github account)
2. [OpenAI account](https://platform.openai.com/signup) (you can sign up through your Github account)

Follow the below steps and provide the **Astra DB API Endpoint**, **Astra DB ApplicationToken** and **OpenAI API Key** when required.

### Sign up for Astra DB
Make sure you have a vector-capable Astra database (get one for free at [astra.datastax.com](https://astra.datastax.com))
- You will be asked to provide the **API Endpoint** which can be found in the right pane underneath *Database details*.
- Ensure you have an **Application Token** for your database which can be created in the right pane underneath *Database details*.

### Sign up for OpenAI
- Create an [OpenAI account](https://platform.openai.com/signup) or [sign in](https://platform.openai.com/login).
- Navigate to the [API key page](https://platform.openai.com/account/api-keys) and create a new **Secret Key**, optionally naming the key.

## Build with Langflow

- If a collection has not yet been created in the Vector database, we need to create a test collection that we can use in this practice. To do this, simply select **Create Collection** from the Data Explorer tab as shown below.

    <p align="left">
    <img  { width=80% } src="./images/3.png">
    </p>


- In AstraUI at [astra.datastax.com](https://astra.datastax.com), click the **Build with Langflow** option as shown below.

- If you don't have a project yet, select **Vector Store RAG** from the **+New Project** option.

    <p align="left"> <img  { width=100% } src="./images/2.png"></p>

- This will bring up an existing **RAG** flow. From this point on, we just need to make the necessary selections.

    <p align="left"> <img  { width=100% } src="./images/4.png"></p>

- First, we make the necessary selections in the flow below to ingest the file we uploaded into the database we created.

    - **Path:** The path where the file to be loaded is located
    - **Model:** The OpenAI model on which the tests will be performed
    - **OpenAI API Key:** A unique identifier that allows users to access OpenAI's models through the API
    - **Collection:** The name of the AstraDB Collection
    - **Astra DB  Application Token:** Used to authenticate applications accessing the DataStax Astra database
    - **Database:** Vector database name

    <p align="left"> <img  { width=100% } src="./images/5.png"></p>

- When we press the play button in the Astra DB box on the right, the data ingestion flow will start and the file we uploaded will be inserted into the Collection in the Database we defined. We can confirm this by going back to the AstraDB interface and checking the collection we created.

    <p align="left"> <img  { width=100% } src="./images/6.png"></p>

- Now, in the flow above, let's define the variables: Text, OpenAI API Key, Database, and Collection. Then, click the pop button in the top right of the Chat Output box. Let's try asking a question to the document we inserted by typing *tell me about something interesting* in the text field to see how the model reacts.

- If there's a green check mark in the upper right corner of the Chat Output box, as shown below, it means the flow has worked without any issues. 

    <p align="left"> <img  { width=100% } src="./images/7.png"></p>

- If we select the Playground option at the bottom left of the page, we can measure the reaction by asking more questions to the inserted data on the screen that appears.

    <p align="left"> <img  { width=100% } src="./images/8.png"></p>


## Creting an API Endpoint

After this step, we will call the created Langflow application using the API endpoint.

1. From the API selection at the bottom right of the page, we select the Python API, activate the Tweaks option, and then copy the Python code from there.

2. We'll make the necessary changes by pasting the copied code into Visual Studio or a similar tool.

3.  Before making changes, you can optionally create a virtual environment. Virtual environments allow you to have a stable, reproducible, and portable environment. You control which package versions are installed and when they are upgraded.
For more information, refer to: [How to create a Virtual Environment](https://github.com/oktytncy/build-rag-chatbot/blob/main/README.md#create-a-virtual-environment-optional)

4. Ensure You Have the Right Environment

    ```bash
    python --version
    ```

    or 

    ```bash
    python3 --version
    ```

- Install Required Libraries: If you haven't already, install the necessary libraries. Open a terminal and run:

    ```bash
    pip install requests argparse langflow
    ```

5. Prepare Your Script
    - Define the **APPLICATION_TOKEN** variable 
    - Delete the **"input_value":  "tell me about something interesting",** line.
    - If you're going to call variables from the operating system like I do, add the line ```import os``` to your Python code.
    - Here's how it should look:
        ```python
        import argparse
        import json
        from argparse import RawTextHelpFormatter
        import requests
        import os
        from typing import Optional
        import warnings
        try:
            from langflow.load import upload_file
        except ImportError:
            warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
            upload_file = None

        BASE_API_URL = "https://api.langflow.astra.datastax.com"
        LANGFLOW_ID = "83fb1d94-ce3c-4308-a098-6efe83048579"
        FLOW_ID = "f89bde0b-a538-498b-945e-d2f62758d0fd"
        ENDPOINT = "" # You can set a specific endpoint name in the flow settings
        APPLICATION_TOKEN = os.getenv('ASTRA_DB_VECTOR_TOKEN')


        # You can tweak the flow by adding a tweaks dictionary
        # e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
        TWEAKS = {
        "ChatInput-ReSAq": {
            "files": "",
            "sender": "User",
            "sender_name": "User",
            "session_id": "",
            "should_store_message": True
        },
        ...
        ```

6. Now, let's run our first test. We will ask the questions in this phase of the application parametrically.

    ``` bash
    % python run_langflow.py "tell me about something interesting"
    ```

- And here is the output;
    ```bash
    {
    "session_id": "f89bde0b-a538-498b-945e-d2f62758d0fd",
    "outputs": [
        {
        "inputs": {
            "input_value": "tell me about something interesting"
        },
        "outputs": [
            {
            "results": {
                "message": {
                "text_key": "text",
                "data": {
                    "text": "The Tesla Model S Plaid is an interesting vehicle due to its remarkable performance capabilities. It boasts the quickest acceleration of any production vehicle, achieving 0-60 mph in just 1.99 seconds. This is made possible by its Tri-Motor All-Wheel Drive system, which maintains over 1,000 horsepower up to a top speed of 200 mph. Additionally, the Model S Plaid features a 22-speaker, 960-watt audio system with Active Road Noise Reduction for an immersive sound experience. Its interior includes a 17-inch touchscreen with high resolution and responsiveness, enhancing the cinematic experience for gaming and movies. The car also supports wireless and 36-watt USB-C charging, ensuring devices stay powered on the go. With over-the-air software updates, the Model S continues to improve over time, offering features like Enhanced Autopilot and Full Self-Driving capabilities.",
                    "sender": "Machine",
                    "sender_name": "AI",
                    "session_id": "f89bde0b-a538-498b-945e-d2f62758d0fd",
                    "files": [],
                    "timestamp": "2024-10-11 14:43:24",
                    "flow_id": "f89bde0b-a538-498b-945e-d2f62758d0fd"
                },
    ```

### Summary

So far, we've quickly built our RAG application with Langflow, ingested our file to the vector database, and successfully asked our first question via the API endpoint. The next step is to make the application more user-friendly.

## Streamlit implementation

First off, to make the Python code cleaner and easier to follow, we move all TWEAKS parameters to the tweaks.json file in the tweaks folder and ensure that the parameters in the code are now read from this file.

**Important :** Changed all True to true, False to false, and None to null, as required by JSON syntax.

```json
{
    "ChatInput-ReSAq": {
      "files": "",
      "sender": "User",
      "sender_name": "User",
      "session_id": "",
      "should_store_message": true
    },
    "AstraVectorStoreComponent-AqKJU": {
      "api_endpoint": "https://19e7aba0-88d0-4e5c-9b13-07019d95061a-us-east-2.apps.astra.datastax.com",
      "batch_size": null,
      "bulk_delete_concurrency": null,
      "bulk_insert_batch_concurrency": null,
      "bulk_insert_overwrite_concurrency": null,
      "collection_indexing_policy": "",
      "collection_name": "my_store",
      "metadata_indexing_exclude": "",
      "metadata_indexing_include": "",
      "metric": "",
      "namespace": "",
      "number_of_results": 4,
      "pre_delete_collection": false,
      "search_filter": {},
      "search_input": "",
      "search_score_threshold": 0,
      "search_type": "Similarity",
      "setup_mode": "Sync",
      "token": "ASTRA_DB_APPLICATION_TOKEN"
    },
    "ParseData-X1paD": {
      "sep": "\n",
      "template": "{text}"
  // Add more components here as needed...
}
```

The updated version of the code should be as follows.

- Make sure to update the LANGFLOW_ID, FLOW_ID, and APPLICATION_TOKEN parameters based on your environment.

```python
import argparse
import json
import requests
import os
import warnings
from argparse import RawTextHelpFormatter
from typing import Optional

# Import upload_file if available
try:
    from langflow.load import upload_file
except ImportError:
    warnings.warn("Langflow provides a function to help you upload files to the flow. Please install langflow to use it.")
    upload_file = None

BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "your_langflow_id"
FLOW_ID = "your_flow_id"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings
APPLICATION_TOKEN = os.getenv('ASTRA_DB_VECTOR_TOKEN')

# Load tweaks from the external tweaks.json file
def load_tweaks():
    tweaks_file_path = os.path.join('tweaks', 'tweaks.json')
    with open(tweaks_file_path, 'r') as file:
        return json.load(file)

def run_flow(message: str,
             endpoint: str,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             application_token: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if application_token:
        headers = {"Authorization": "Bearer " + application_token, "Content-Type": "application/json"}
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="""Run a flow with a given message and optional tweaks.
Run it like: python <your file>.py "your message here" --endpoint "your_endpoint" --tweaks '{"key": "value"}'""",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("message", type=str, help="The message to send to the flow")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT or FLOW_ID, help="The ID or the endpoint name of the flow")
    parser.add_argument("--tweaks", type=str, help="JSON string representing the tweaks to customize the flow", default=None)
    parser.add_argument("--application_token", type=str, default=APPLICATION_TOKEN, help="Application Token for authentication")
    parser.add_argument("--output_type", type=str, default="chat", help="The output type")
    parser.add_argument("--input_type", type=str, default="chat", help="The input type")
    parser.add_argument("--upload_file", type=str, help="Path to the file to upload", default=None)
    parser.add_argument("--components", type=str, help="Components to upload the file to", default=None)

    args = parser.parse_args()

    # Load tweaks from file if --tweaks is not provided
    if args.tweaks:
        try:
            tweaks = json.loads(args.tweaks)
        except json.JSONDecodeError:
            raise ValueError("Invalid tweaks JSON string")
    else:
        tweaks = load_tweaks()

    if args.upload_file:
        if not upload_file:
            raise ImportError("Langflow is not installed. Please install it to use the upload_file function.")
        elif not args.components:
            raise ValueError("You need to provide the components to upload the file to.")
        tweaks = upload_file(file_path=args.upload_file, host=BASE_API_URL, flow_id=ENDPOINT, components=args.components, tweaks=tweaks)

    response = run_flow(
        message=args.message,
        endpoint=args.endpoint,
        output_type=args.output_type,
        input_type=args.input_type,
        tweaks=tweaks,
        application_token=args.application_token
    )

    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
```

- One of the key reasons for simplifying this way is that we can configure the RAG application by just updating the tweaks.json file, without needing to modify the code itself.

1. In order to perform streamlit integration, the necessary library must first be installed.

    ```python
    pip install streamlit
    ```

2. To use streamlit, update the code as follows and update the run_flow defining function while doing so.

    ```python
    import streamlit as st

    # Function to run flow with Streamlit inputs
    def run_flow(message: str, tweaks: dict, endpoint: str = "your-endpoint", output_type: str = "chat", input_type: str = "chat"):
        BASE_API_URL = "https://api.langflow.astra.datastax.com"
        LANGFLOW_ID = "your_langflow_id"
        api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

        payload = {
            "input_value": message,
            "output_type": output_type,
            "input_type": input_type,
            "tweaks": tweaks
        }

        application_token = os.getenv('ASTRA_DB_VECTOR_TOKEN')
        headers = {"Authorization": f"Bearer {application_token}", "Content-Type": "application/json"}
        response = requests.post(api_url, json=payload, headers=headers)
        return response.json()

    # Streamlit app setup
    st.title('LangFlow Chat Application')

    # User input text box
    user_message = st.text_input('Enter your message')

    # Display tweaks (for demonstration purposes)
    tweaks = load_tweaks()
    st.write("Tweaks: ", tweaks)

    # When the user submits a message, call the API
    if st.button('Send'):
        if user_message:
            st.write(f"Sending message: {user_message}")
            response = run_flow(user_message, tweaks=tweaks)
            # Display the response from the API
            st.write("Response: ", response)
        else:
            st.write("Please enter a message.")
    ```

3. This part of the code still doesn't generate good output, and the parameters aren't flexible. To make it more user-friendly, we're making some key parameters adjustable.

    ```python
    import streamlit as st
    import json
    import os
    import requests

    # Load tweaks from file
    def load_tweaks():
        tweaks_file_path = os.path.join('tweaks', 'tweaks.json')
        with open(tweaks_file_path, 'r') as file:
            return json.load(file)

    # Function to run flow with Streamlit inputs
    def run_flow(message: str, tweaks: dict, endpoint: str = "your_flow_id", output_type: str = "chat", input_type: str = "chat"):
        BASE_API_URL = "https://api.langflow.astra.datastax.com"
        LANGFLOW_ID = "your_langflow_id"
        api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{endpoint}"

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
            response = run_flow(user_message, tweaks=tweaks)
            
            # Extract and display only the message part
            extracted_message = extract_message(response)
            st.write("Response: ", extracted_message)
        else:
            st.write("Please enter a message.")
    ```

    - You can check out the final version of the application below.

    ```python
    streamlit run app.py
    ```

    <p align="left"> <img  { width=100% } src="./images/9.png"></p>

4. The next step will be to read the langflow_id and flow_id parameters from a config file instead of hardcoding them in the code. Create a folder named conf and place a config.json file inside it with the following structure:

    ```json
    {
    "langflow_id": "your_langflow_id",
    "flow_id": "your_flow_id"
    }
    ```

- Here’s the updated code that loads langflow_id and flow_id from the config.json file:

    ```python
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

    # Configurable chunk_overlap (slider)
    chunk_overlap = st.sidebar.slider('Chunk Overlap', min_value=0, max_value=500, value=tweaks.get('SplitText-EmpzC', {}).get('chunk_overlap', 200))
    tweaks['SplitText-EmpzC']['chunk_overlap'] = chunk_overlap

    # Configurable chunk_size (slider)
    chunk_size = st.sidebar.slider('Chunk Size', min_value=500, max_value=2000, value=tweaks.get('SplitText-EmpzC', {}).get('chunk_size', 1000))
    tweaks['SplitText-EmpzC']['chunk_size'] = chunk_size

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
    ```