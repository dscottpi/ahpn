from sqlalchemy import make_url

from ahpn.config import DEFAULT_DATABASE_URL
from ahpn.db import database_url_from_env


def test_database_url_prefers_full_url() -> None:
    env = {"AHPN_DATABASE_URL": "postgresql+psycopg://a:b@h/db", "AHPN_DB_HOST": "ignored"}
    assert database_url_from_env(env) == "postgresql+psycopg://a:b@h/db"


def test_database_url_defaults_to_local_compose() -> None:
    assert database_url_from_env({}) == DEFAULT_DATABASE_URL


def test_database_url_from_parts_escapes_password() -> None:
    env = {
        "AHPN_DB_HOST": "db.example.rds.amazonaws.com",
        "AHPN_DB_USER": "ahpn",
        "AHPN_DB_PASSWORD": "p@ss/w:rd#1",
        "AHPN_DB_NAME": "ahpn",
    }
    url = make_url(database_url_from_env(env))

    assert url.host == "db.example.rds.amazonaws.com"
    assert url.port == 5432
    assert url.password == "p@ss/w:rd#1"
    assert url.database == "ahpn"
