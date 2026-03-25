from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import yaml

from src.api.routes import tasks, products, standards, wiring
from src.database import engine, Base
from src.utils.config import load_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="接线视觉检测系统", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router, prefix="/api/tasks", tags=["任务管理"])
app.include_router(products.router, prefix="/api/products", tags=["产品管理"])
app.include_router(standards.router, prefix="/api/standards", tags=["标准图管理"])
app.include_router(wiring.router, prefix="/api/wiring", tags=["接线表管理"])

app.mount("/static", StaticFiles(directory="web"), name="static")
app.mount("/exports", StaticFiles(directory="data/exports"), name="exports")
app.mount("/standards", StaticFiles(directory="data/standard_images"), name="standards")


@app.get("/")
async def root():
    return {"message": "接线视觉检测系统 API", "version": "1.0.0"}


@app.get("/api/stations")
async def get_stations():
    with open("config/station_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["stations"]


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}
