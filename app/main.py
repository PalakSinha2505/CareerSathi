from fastapi import FastAPI
from app.api.routes import interview, feedback
from app.db.database import engine
from app.db.models import Base
from app.api.routes import auth, user

app = FastAPI(
    title="Mock Interview AI",
    version="1.0.0"
)

app.include_router(interview.router, prefix="/interview", tags=["Interview"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])


@app.get("/")
def root():
    return {"message": "Mock Interview API is running"}

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(user.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

