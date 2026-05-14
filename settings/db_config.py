"""Database configuration utilities.

All credentials are loaded from environment variables (optionally via a .env file).
This module centralizes database connection handling for ETL and ML scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

from dotenv import load_dotenv


ENV_DB_HOST: str = "DB_HOST"
ENV_DB_PORT: str = "DB_PORT"
ENV_DB_USER: str = "DB_USER"
ENV_DB_PASSWORD: str = "DB_PASSWORD"
ENV_DB_NAME: str = "DB_NAME"


@dataclass(frozen=True)
class DBConfig:
    """Typed database configuration."""

    host: str
    port: int
    user: str
    password: str
    dbname: str


def load_environment() -> None:
    """Load environment variables from a local .env file if present."""

    load_dotenv(override=False)


def get_db_config() -> DBConfig:
    """Read DB configuration from environment variables.

    Returns:
        DBConfig: Database connection parameters.

    Raises:
        ValueError: If required environment variables are missing.
    """

    load_environment()

    missing = [
        name
        for name in [
            ENV_DB_HOST,
            ENV_DB_PORT,
            ENV_DB_USER,
            ENV_DB_PASSWORD,
            ENV_DB_NAME,
        ]
        if not os.getenv(name)
    ]
    if missing:
        raise ValueError(
            "Missing required DB environment variables: "
            + ", ".join(missing)
            + ". Create a .env file or export them in your shell."
        )

    return DBConfig(
        host=str(os.environ[ENV_DB_HOST]),
        port=int(os.environ[ENV_DB_PORT]),
        user=str(os.environ[ENV_DB_USER]),
        password=str(os.environ[ENV_DB_PASSWORD]),
        dbname=str(os.environ[ENV_DB_NAME]),
    )


def to_sqlalchemy_url(cfg: DBConfig) -> str:
    """Build a SQLAlchemy PostgreSQL connection URL.

    Args:
        cfg: Database configuration.

    Returns:
        A SQLAlchemy URL string.
    """

    return (
        f"postgresql+psycopg2://{cfg.user}:{cfg.password}"
        f"@{cfg.host}:{cfg.port}/{cfg.dbname}"
    )


def as_dict(cfg: DBConfig) -> Dict[str, str]:
    """Convert config to a dict for libraries that expect kwargs."""

    return {
        "host": cfg.host,
        "port": str(cfg.port),
        "user": cfg.user,
        "password": cfg.password,
        "dbname": cfg.dbname,
    }
