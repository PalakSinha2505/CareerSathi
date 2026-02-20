from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas.user import UserCreate, UserLogin
from app.auth.auth_service import register_user, login_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, db: Session = Depends(get_db)):
    created_user = register_user(
        db=db,
        name=user.name,
        email=user.email,
        password=user.password
    )

    token = login_user(
        db=db,
        email=user.email,
        password=user.password
    )

    return {
        "user": {
            "id": created_user.id,
            "name": created_user.name,
            "email": created_user.email,
        },
        "access_token": token,
        "token_type": "bearer"
    }



@router.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user, token = login_user(
        db=db,
        email=user.email,
        password=user.password
    )

    return {
        "user": {
            "id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
        },
        "access_token": token,
        "token_type": "bearer"
    }

