from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import os

# Load environment variables from .env file
load_dotenv()

# Access the database credentials
"""db_host = os.getenv('DB_HOST')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')"""

# SQLAlchemy setup
DATABASE_URL = "sqlite:////RAG_stack/Integrated/database/chat_database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session=SessionLocal()

# Create base class for models
Base = declarative_base()

# Define the ChatContext model
class ChatContext(Base):
    __tablename__ = 'chat_context'
    
    username = Column(String, primary_key=True)
    context = Column(String)









