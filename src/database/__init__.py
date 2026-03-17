from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import yaml

with open("config/database.yaml", "r", encoding="utf-8") as f:
    db_config = yaml.safe_load(f)

env = "development"
DB_URL = (
    f"postgresql://{db_config[env]['username']}:{db_config[env]['password']}"
    f"@{db_config[env]['host']}:{db_config[env]['port']}/{db_config[env]['database']}"
)

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
