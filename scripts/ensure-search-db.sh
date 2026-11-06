#!/usr/bin/env bash
# Install pgvector (local apt Postgres) and initialize the AHPN search schema.
set -euo pipefail

PG_VER="$(ls /etc/postgresql 2>/dev/null | sort -V | tail -1 || true)"
if [[ -n "${PG_VER}" ]]; then
  if ! dpkg -s "postgresql-${PG_VER}-pgvector" >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "postgresql-${PG_VER}-pgvector"
  fi
  sudo -u postgres psql -p 5433 -d ahpn -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
ALTER ROLE ahpn WITH SUPERUSER;
SQL
  # Optional isolated DB for integration tests.
  if ! sudo -u postgres psql -p 5433 -tAc "SELECT 1 FROM pg_database WHERE datname='ahpn_test'" | grep -q 1; then
    sudo -u postgres psql -p 5433 -c "CREATE DATABASE ahpn_test OWNER ahpn;"
  fi
  sudo -u postgres psql -p 5433 -d ahpn_test -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
GRANT ALL ON SCHEMA public TO ahpn;
SQL
fi

uv run ahpn db-init
