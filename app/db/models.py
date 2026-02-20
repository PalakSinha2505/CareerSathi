from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from app.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Interview(Base):
    __tablename__ = "interviews"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String, nullable=False)
    level = Column(String, nullable=False)
    score = Column(Integer)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="interviews")
    answers = relationship("QuestionAnswer", back_populates="interview", cascade="all, delete-orphan")

class QuestionAnswer(Base):
    __tablename__ = "question_answers"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id"))
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    analysis = Column(Text)     
    feedback = Column(Text)      
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    interview = relationship("Interview", back_populates="answers")    