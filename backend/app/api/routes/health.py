"""Health-check API route."""

from fastapi import APIRouter

from backend.app.core.config import settings


router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok", "service": settings.app_name}
