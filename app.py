import streamlit as st
import os
from typing import List, Dict, Any, Optional
from llmhandler import get_llm_from_model

# Set page configuration
st.set_page_config(
    page_title="LLM Chat with llm-handler",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = None

# App title and description
st.title("🤖 LLM Chat Application")
st.markdown("""
This application demonstrates how to use the llm-handler library to create a chat interface
with various LLM providers including OpenAI, Anthropic, Google, and more.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    model_options = {
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-haiku-3.5"],
        "Google": ["gemini-2.0-pro", "gemini-2.0-flash"],
        "Cohere": ["command", "command-light"]
    }
    
    provider = st.selectbox("Provider", list(model_options.keys()))
    model = st.selectbox("Model", model_options[provider])
    
    # API key input
    api_key = st.text_input("API Key", type="password")
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value="You are a helpful, friendly AI assistant. Provide clear and concise responses.",
        height=150
    )
    
    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Max tokens slider
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
    
    # Initialize or update LLM when settings change
    if st.button("Apply Settings") or st.session_state.llm is None:
        if api_key:
            with st.spinner("Initializing LLM..."):
                try:
                    st.session_state.llm = get_llm_from_model(model, api_key=api_key)
                    st.success(f"Successfully initialized {model}")
                except Exception as e:
                    st.error(f"Error initializing LLM: {str(e)}")
        else:
            st.warning("Please enter an API key")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if LLM is initialized
    if st.session_state.llm is None:
        with st.chat_message("assistant"):
            st.error("Please configure and initialize the LLM in the sidebar first.")
    else:
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Format messages for the LLM
            formatted_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    formatted_messages.append({
                        "message_type": "human",
                        "message": msg["content"],
                        "image_paths": [],
                        "image_urls": []
                    })
                elif msg["role"] == "assistant":
                    formatted_messages.append({
                        "message_type": "ai",
                        "message": msg["content"]
                    })
            
            # Stream the response
            try:
                # Define callback for streaming
                def process_chunk(chunk, usage_data):
                    nonlocal full_response
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Stream generate
                for _, _ in st.session_state.llm.stream_generate(
                    event_id="streamlit_chat",
                    system_prompt=system_prompt,
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temp=temperature,
                    callback=process_chunk
                ):
                    pass
                
                # Display final response
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add a button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display usage information in the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses the [llm-handler](https://github.com/yourusername/llm-handler) library 
    to provide a unified interface for interacting with various LLM providers.
    
    Features:
    - Support for multiple providers (OpenAI, Anthropic, Google, Cohere)
    - Streaming responses
    - Configurable parameters (temperature, max tokens)
    - Custom system prompts
    """)