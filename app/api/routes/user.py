from fastapi import APIRouter, Depends
from app.auth.dependencies import get_current_user
from app.db.models import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
def read_current_user(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email
    }
