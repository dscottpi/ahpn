from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ahpn.config import DEFAULT_DATABASE_URL
from ahpn.models import Base


def database_url_from_env(env: Mapping[str, str] | None = None) -> str:
    """AHPN_DATABASE_URL, else a URL built from AHPN_DB_* parts, else the local Compose default.

    The parts form exists for ECS, where the RDS-managed password arrives as
    its own secret and may contain characters that need escaping in a URL.
    """
    env = os.environ if env is None else env
    if env.get("AHPN_DATABASE_URL"):
        return env["AHPN_DATABASE_URL"]
    host = env.get("AHPN_DB_HOST")
    if not host:
        return DEFAULT_DATABASE_URL
    url = URL.create(
        "postgresql+psycopg",
        username=env.get("AHPN_DB_USER", "ahpn"),
        password=env.get("AHPN_DB_PASSWORD"),
        host=host,
        port=int(env.get("AHPN_DB_PORT", "5432")),
        database=env.get("AHPN_DB_NAME", "ahpn"),
    )
    return url.render_as_string(hide_password=False)


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def init_db(database_url: str) -> None:
    engine = create_db_engine(database_url)
    Base.metadata.create_all(engine)
    engine.dispose()


def session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(database_url), expire_on_commit=False)
