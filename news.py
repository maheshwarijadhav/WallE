
# this cde is jut to sow to man 
# Bert

import streamlit as st
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import logging
from langchain_together import Together
from config import TOGETHER_API_KEY
import os
from langchain.prompts import PromptTemplate
import re

# Set up Together API Key
os.environ['bert'] = TOGETHER_API_KEY

# Create a logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Function to load a subset of product data
def load_product_data(file_path, nrows=10000):
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        st.error("Failed to load product data. Please check the file and try again.")
        return None

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error("Failed to create vector store. Please try again.")
        return None

# Function to generate Walmart product links
def generate_walmart_link(product_name):
    base_url = "https://www.walmart.com"
    search_url = f"{base_url}/search?q={product_name.replace(' ', '+')}"
    return search_url

# Define the prompt template
prompt_template = """
<s>[INST]
As Wall-E, Walmart's AI Shopping Assistant, your task is to provide helpful and accurate responses to customer queries about Walmart products. Follow these guidelines:

- Respond in a friendly, concise manner that reflects Walmart's customer service standards.
- Provide specific product recommendations based on the customer's query and the available product data.
- For each product recommendation, include a Walmart product link using the following format: [Product Name](product_link)
- If asked about prices or availability, give the most up-to-date information from the product data.
- Offer alternatives if a specific product is not available or if there might be better options for the customer.
- If you're unsure about any information, politely say so and offer to help find more details.
- Conclude each response with a question or suggestion to encourage further engagement with the customer.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
CUSTOMER QUERY: {question}
WALL-E'S RESPONSE:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Function to create a conversational chain using Together API
def get_conversational_chain(vector_store):
    try:
        llm = Together(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1024,
            top_k=50,
            top_p=0.95,
            together_api_key=os.getenv('TOGETHER_API_KEY')
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {str(e)}")
        st.error("Failed to initialize the AI model. Please try again.")
        return None

# Function to handle user input and process the response
def user_input(user_question, conversation):
    try:
        response = conversation({'question': user_question})
        chat_history = response['chat_history']
        
        # Process the AI's response to include Walmart links
        ai_response = chat_history[-1].content
        processed_response = re.sub(r'\[([^\]]+)\]', lambda m: f"[{m.group(1)}]({generate_walmart_link(m.group(1))})", ai_response)
        
        chat_history[-1].content = processed_response
        return chat_history
    except Exception as e:
        logger.exception("Error in user input processing")
        st.error(f"An error occurred: {str(e)}")
        return []

def main():
    st.set_page_config(page_title="Wall-E: Walmart's AI Shopping Assistant", layout="wide")
    st.header("Wall-E: Your Walmart Shopping Assistant 🛒")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.chat_history = []
        
        # Load and process the product data
        with st.spinner("Loading and processing product data..."):
            product_data = load_product_data("Apple.csv")
            if product_data:
                text_chunks = get_text_chunks(product_data)
                vector_store = get_vector_store(text_chunks)
                if vector_store:
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Product data loaded successfully. You can start chatting now!")
                else:
                    st.error("Failed to process the product data. Please check the file and restart the application.")

    # Main chat interface
    user_question = st.chat_input("What are you looking for today?")
    
    if user_question:
        if st.session_state.conversation:
            st.session_state.chat_history = user_input(user_question, st.session_state.conversation)  # Update chat history
        else:
            st.warning("The AI model is not initialized. Please check the product data and restart the application.")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)  # Use st.markdown to render the links correctly

if __name__ == "__main__":
    main()