import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import yaml

config_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "config", "database.yaml"
)

if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        db_config = yaml.safe_load(f)
    env = "development"
    DB_URL = (
        f"postgresql://{db_config[env]['username']}:{db_config[env]['password']}"
        f"@{db_config[env]['host']}:{db_config[env]['port']}/{db_config[env]['database']}"
    )
else:
    DB_URL = "sqlite:///linedet.db"

use_sqlite = os.environ.get("USE_SQLITE", "true").lower() == "true"

if use_sqlite:
    engine = create_engine(
        "sqlite:///linedet.db",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(DB_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
