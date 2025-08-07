from fastapi import FastAPI, Depends, HTTPException
from pydanticmodel import QueryRequest
from RAG import query_agent
from sqlalchemy.orm import Session
from database.database import SessionLocal,ChatContext

def get_db():
    db_session = SessionLocal()  # Create a new session for each request
    try:
        yield db_session
    finally:
        db_session.close()  # Ensure the session is closed when done

# Create the FastAPI app
app = FastAPI()

@app.post("/chat")
async def chat(request: QueryRequest, db: Session = Depends(get_db)):
    # Use db passed through dependency injection
    result = query_agent(request.username, request.query, db)
    return result

