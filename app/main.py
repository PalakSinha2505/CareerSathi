from fastapi import FastAPI
from app.api.routes import interview, feedback

app = FastAPI(
    title="Mock Interview AI",
    version="1.0.0"
)

app.include_router(interview.router, prefix="/interview", tags=["Interview"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])


@app.get("/")
def root():
    return {"message": "Mock Interview API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

