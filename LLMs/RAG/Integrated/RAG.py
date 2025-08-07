import os
import sqlite3
from langchain_huggingface import HuggingFaceEndpoint # type: ignore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from vectorstore.vectordbsetup import vector_db  # Import the vector database
from database.database import ChatContext,db_session
from huggingface_hub import login
from sqlalchemy.orm import Session

HF_TOKEN = os.environ.get("HF_TOKEN")
login(HF_TOKEN)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the LLM (HuggingFace model)
def load_llm(huggingface_repo_id):

    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Custom Prompt Template for the agent
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {query}

Start the answer directly. No small talk please.
"""

# Set up the custom prompt
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "query"])

# Create QA chain
def create_qa_chain():
    llm = load_llm(HUGGINGFACE_REPO_ID)  # Load the LLM (e.g., HuggingFace model)
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Retrieve chat context from the database
def get_chat_context(username: str, db_session):
    try:
        # Fetch the chat context using username as the primary key
        chat_context = db_session.query(ChatContext).filter(ChatContext.username == username).one()
        return chat_context.context
    except Exception:
        # If no record is found, return an empty context
        return ""

# Update the chat context in the database
def update_chat_context(username: str, new_context: str, db_session):
    chat_context = db_session.query(ChatContext).filter(ChatContext.username == username).first()
    if chat_context:
        # If the context exists, update it
        chat_context.context = new_context
    else:
        # If no context exists, create a new one
        chat_context = ChatContext(username=username, context=new_context)
        db_session.add(chat_context)
    db_session.commit()  # Commit changes to the database

# Invoke the agent with a query and update the context
def query_agent(username: str, query: str, db_session: Session):
    # Retrieve previous chat context for the user
    context = get_chat_context(username, db_session)

    # Create the QA chain
    qa_chain = create_qa_chain()

    # Prepare the prompt and invoke the agent
    response = qa_chain.invoke({'context': context, 'query': query})

    # Append the new query and response to the context
    new_context = f"{context}\nUser: {query}\nAgent: {response['result']}"

    # Update the chat context in the database
    update_chat_context(username, new_context, db_session)

    # Return the response and updated context
    return {"response": response['result'], "context": new_context}

db_session.close()

