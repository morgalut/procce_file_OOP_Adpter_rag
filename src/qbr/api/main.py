"""FastAPI ASGI entry point: python -m qbr.api or uvicorn qbr.api.main:app."""
from fastapi import FastAPI
from qbr.api.routes import build_api_router
from qbr.bootstrap.runtime import api_lifespan

def create_app() -> FastAPI:
    app = FastAPI(title='QBR Intelligence API', version='0.27.0', lifespan=api_lifespan)
    app.include_router(build_api_router())
    return app

app = create_app()
