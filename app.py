"""Gradio web interface for the Icebreaker Bot."""

import os
import sys
import logging
import uuid
import gradio as gr

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database
from modules.llm_interface import change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query
import os
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Dictionary to store active conversations
active_indices = {}

def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and generate initial facts.
    
    Args:
        linkedin_url: LinkedIn profile URL to process.
        api_key: ProxyCurl API key.
        use_mock: Whether to use mock data.
        selected_model: LLM model to use.
        
    Returns:
        Initial facts about the profile and a session ID for this conversation.
    """
    # For the starter template, we'll return helpful messages
    # This will be replaced with actual implementation
    
    # Generate a mock session ID for demonstration purposes
    mock_session_id = str(uuid.uuid4())
    
    if use_mock:
        return (
            "✅ Mock data selected! When implemented, this function will:\n\n"
            "1. Load mock LinkedIn data from a pre-made file\n"
            "2. Split the data into nodes\n"
            "3. Create a vector database\n"
            "4. Generate interesting facts about the profile\n\n"
            "TO DO: Implement the process_profile function to make this work!",
            mock_session_id
        )
    elif not linkedin_url:
        return "⚠️ Please enter a LinkedIn profile URL or select 'Use Mock Data'.", None
    elif not api_key and not use_mock:
        return "⚠️ Please enter a ProxyCurl API key or select 'Use Mock Data'.", None
    else:
        if selected_model != config.LLM_MODEL_ID:
            model_msg = f"\nWhen implemented, will use model: {selected_model}"
        else:
            model_msg = ""
            
        return (
            f"⏳ Function not yet implemented. When completed, this will:\n\n"
            f"1. Process the LinkedIn profile at: {linkedin_url}\n"
            f"2. Extract and analyze the profile data\n"
            f"3. Generate interesting facts about this person's career{model_msg}\n\n"
            f"TO DO: Implement the process_profile function to make this work!",
            mock_session_id
        )

def chat_with_profile(session_id, user_query, chat_history):
    """Chat with a processed LinkedIn profile.
    
    Args:
        session_id: Session ID for this conversation.
        user_query: User's question.
        chat_history: Chat history.
        
    Returns:
        Updated chat history.
    """
    # For the starter template, we'll return helpful messages
    # This will be replaced with actual implementation
    
    if not session_id:
        return chat_history + [[user_query, "⚠️ No profile loaded. Please process a LinkedIn profile first."]]
    elif not user_query.strip():
        return chat_history + [["", "⚠️ Please enter a question."]]
    else:
        return chat_history + [[
            user_query, 
            f"⏳ When implemented, this function will answer your question: '{user_query}'\n\n"
            f"TO DO: Implement the chat_with_profile function to make this work!"
        ]]

def create_gradio_interface():
    """Create the Gradio interface for the Icebreaker Bot."""
    # Define available LLM models
    available_models = [
        "ibm/granite-3-2-8b-instruct",
        "meta-llama/llama-3-3-70b-instruct"
    ]
    
    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles")
        
        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/"
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (Leave empty to use mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password"
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select LLM Model",
                        value=config.LLM_MODEL_ID
                    )
                    process_btn = gr.Button("Process Profile")
                
                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)
            
            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id]
            )
        
        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")
            
            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?"
            )
            
            chat_btn = gr.Button("Send")
            
            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
            
            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot]
            )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    # Launch the Gradio interface
    # You can customize these parameters:
    # - share=True creates a public link you can share with others
    # - server_name and server_port set where the app runs
    demo.launch(
        server_name="127.0.0.1",  
        server_port=5000,
        share=True  # Set to False if you don't want to create a public link
    )