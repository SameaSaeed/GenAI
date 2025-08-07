from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# SQLAlchemy setup
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define the ChatContext model
class ChatContext(Base):
    __tablename__ = 'chat_context'
    
    username = Column(String, primary_key=True)
    context = Column(String)

# Create the database table
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    context: str

class UserResponse(BaseModel):
    username: str
    context: str

    class Config:
        orm_mode = True  # Required for compatibility with SQLAlchemy models

# FastAPI app setup
app = FastAPI()

# Dependency for getting the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Endpoint to process queries and store them in the database
@app.post("/process_query/")
def process_query(user_name: str, query: str, db: Session = Depends(get_db)):
    # Logic to process the query and prepare the response
    response = f"Processed query for {user_name}: {query}"
    
    # Check if the user already exists in the chat_context table
    existing_user = db.query(ChatContext).filter(ChatContext.username == user_name).first()
    
    if existing_user:
        # Append the new query and response to the existing context
        existing_user.context += f"\nQuery: {query}\nResponse: {response}"
    else:
        # Create a new entry for the user if not already present
        new_user = ChatContext(username=user_name, context=f"Query: {query}\nResponse: {response}")
        db.add(new_user)
    
    db.commit()  # Commit the changes to the database
    db.refresh(existing_user if existing_user else new_user)  # Refresh to get updated data

    return UserResponse(username=new_user.username, context=new_user.context)

