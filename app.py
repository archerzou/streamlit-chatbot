import logging
import os
import streamlit as st
from model_serving_utils import query_endpoint, is_endpoint_supported
from ai_query_utils import query_impairment_data, get_table_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure environment variable is set correctly
SERVING_ENDPOINT = os.getenv('SERVING_ENDPOINT')
assert SERVING_ENDPOINT, \
    ("Unable to determine serving endpoint to use for chatbot app. If developing locally, "
     "set the SERVING_ENDPOINT environment variable to the name of your serving endpoint. If "
     "deploying to a Databricks app, include a serving endpoint resource named "
     "'serving_endpoint' with CAN_QUERY permissions, as described in "
     "https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app#deploy-the-databricks-app")

# AI Query configuration
AI_QUERY_ENDPOINT = "databricks-gpt-oss-120b"
CATALOG = "dev_structured"
SCHEMA = "analytics"
TABLE = "measuresponses_impairment"

# Check if the endpoint is supported
endpoint_supported = is_endpoint_supported(SERVING_ENDPOINT)

def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_info = get_user_info()

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.title("Healthcare Impairment Data Assistant")

# Check if endpoint is supported and show appropriate UI
if not endpoint_supported:
    st.error("Unsupported Endpoint Type")
    st.markdown(
        f"The endpoint `{SERVING_ENDPOINT}` is not compatible with this basic chatbot template.\n\n"
        "This template only supports chat completions-compatible endpoints.\n\n"
        "For a richer chatbot template that supports all conversational endpoints on Databricks, "
        "please see the [Databricks documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app)."
    )
else:
    # Mode selector in sidebar
    with st.sidebar:
        st.header("Query Mode")
        query_mode = st.radio(
            "Select how to interact:",
            options=["Data Query (ai_query)", "General Chat"],
            index=0,
            help="Data Query uses ai_query to answer questions about the impairment data table. General Chat uses the standard chat endpoint."
        )
        
        if query_mode == "Data Query (ai_query)":
            st.info(f"**Connected to:**\n- Table: `{CATALOG}.{SCHEMA}.{TABLE}`\n- Endpoint: `{AI_QUERY_ENDPOINT}`")
            
            with st.expander("Table Information"):
                st.markdown(get_table_info())
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Main chat area
    if query_mode == "Data Query (ai_query)":
        st.markdown(
            "Ask questions about the **healthcare impairment data** in natural language. "
            "Your questions will be processed using Databricks `ai_query` function."
        )
    else:
        st.markdown(
            "General chat mode. See "
            "[Databricks docs](https://docs.databricks.com/aws/en/generative-ai/agent-framework/chat-app) "
            "for more information."
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    placeholder_text = "Ask about impairment data..." if query_mode == "Data Query (ai_query)" else "What is up?"
    if prompt := st.chat_input(placeholder_text):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if query_mode == "Data Query (ai_query)":
                # Use ai_query to process the question
                with st.spinner("Querying impairment data..."):
                    result = query_impairment_data(
                        user_question=prompt,
                        mode="general",
                        endpoint=AI_QUERY_ENDPOINT,
                        catalog=CATALOG,
                        schema=SCHEMA,
                        table=TABLE
                    )
                
                if result["success"]:
                    assistant_response = result["response"]
                    if isinstance(assistant_response, list):
                        # Format analysis results
                        from ai_query_utils import format_analysis_response
                        assistant_response = format_analysis_response(assistant_response)
                else:
                    assistant_response = f"Error querying data: {result.get('error', 'Unknown error')}"
            else:
                # Use standard chat endpoint
                assistant_response = query_endpoint(
                    endpoint_name=SERVING_ENDPOINT,
                    messages=st.session_state.messages,
                    max_tokens=400,
                )["content"]
            
            st.markdown(assistant_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
