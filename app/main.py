from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import interview, feedback, auth, user
from app.db.database import engine
from app.db.models import Base

app = FastAPI(
    title="Mock Interview AI",
    version="1.0.0"
)

# ---------------- CORS CONFIG ----------------
origins = [
    "http://localhost:5173",  # Vite dev
    "https://careersathi-rm5f.onrender.com",  # production frontend (if hosted)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- ROUTES ----------------
app.include_router(interview.router)
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
app.include_router(auth.router)
app.include_router(user.router)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Mock Interview API is running"}

# ---------------- DB INIT ----------------
Base.metadata.create_all(bind=engine)

# ---------------- LOCAL RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)