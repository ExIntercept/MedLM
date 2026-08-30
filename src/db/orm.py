"""SQLModel database engine and session generator."""
import os
from sqlmodel import Session, SQLModel, create_engine
from src.config import DATA_DIR

DB_PATH = DATA_DIR / "medical_history.db"

ENGINE = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


def init_db() -> None:
    from src.db import models  # noqa: F401
    SQLModel.metadata.create_all(ENGINE)


def get_db_session():
    with Session(ENGINE) as session:
        yield session
