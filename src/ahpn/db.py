from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ahpn.models import Base


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def init_db(database_url: str) -> None:
    engine = create_db_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()


def session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(database_url), expire_on_commit=False)
