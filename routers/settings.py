"""
RAGSmith – Settings router
Manage application settings including API keys (Groq, S3, etc.)
with secure storage and validation.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from database import get_connection, db_fetchone, db_execute, db_insert, ph
from config import get_settings

router = APIRouter()
logger = logging.getLogger("ragsmith.settings")


# ── Models ────────────────────────────────────────────────────────────────────

class SettingUpdate(BaseModel):
    groq_api_key: Optional[str] = Field(None, description="Groq API key (will be masked in responses)")
    s3_bucket_name: Optional[str] = None
    s3_region: Optional[str] = None


class SettingResponse(BaseModel):
    groq_api_key_configured: bool
    s3_bucket_name: Optional[str]
    s3_region: Optional[str]
    llm_provider: str
    available_models: list[str]


class ModelListResponse(BaseModel):
    provider: str
    models: list[str]


class APIKeyValidationRequest(BaseModel):
    api_key: str = Field(..., description="API key to validate")


class APIKeyValidationResponse(BaseModel):
    valid: bool
    message: str
    provider: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_setting(key: str, value: str) -> None:
    """Save a setting to the database (create or update)."""
    conn = get_connection()
    try:
        existing = db_fetchone(conn, f"SELECT value FROM app_settings WHERE key={ph()}", (key,))
        if existing:
            db_execute(conn,
                f"UPDATE app_settings SET value={ph()}, updated_at=datetime('now') WHERE key={ph()}",
                (value, key), commit=True)
        else:
            db_insert(conn,
                f"INSERT INTO app_settings (key, value) VALUES ({ph()},{ph()})",
                (key, value), commit=True)
        logger.debug("Setting saved: %s", key)
    finally:
        conn.close()


def _load_setting(key: str) -> Optional[str]:
    """Load a setting from the database."""
    conn = get_connection()
    try:
        row = db_fetchone(conn, f"SELECT value FROM app_settings WHERE key={ph()}", (key,))
        return row["value"] if row else None
    finally:
        conn.close()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/", response_model=SettingResponse)
def get_settings_info():
    """Get current settings configuration and available models."""
    cfg = get_settings()
    
    # Load stored settings
    groq_key = _load_setting("groq_api_key") or cfg.groq_api_key
    s3_bucket = _load_setting("s3_bucket_name") or cfg.s3_bucket_name
    s3_region = _load_setting("s3_region") or cfg.aws_region
    
    # Get available models based on provider
    if cfg.llm_provider == "groq":
        # Parse from config
        models = [m.strip() for m in cfg.groq_available_models.split(",")]
    else:
        models = [m.strip() for m in cfg.ollama_available_models.split(",")]
    
    return SettingResponse(
        groq_api_key_configured=bool(groq_key),
        s3_bucket_name=s3_bucket,
        s3_region=s3_region,
        llm_provider=cfg.llm_provider,
        available_models=models,
    )


@router.post("/groq/validate", response_model=APIKeyValidationResponse)
def validate_groq_api_key(body: APIKeyValidationRequest):
    """Validate Groq API key by attempting a test call."""
    if not body.api_key or not body.api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="API key cannot be empty"
        )
    
    try:
        from services.llm import _check_groq, _groq_list_models
        
        is_valid = _check_groq(body.api_key)
        
        if is_valid:
            # Try to list models to ensure full access
            models = _groq_list_models(body.api_key)
            return APIKeyValidationResponse(
                valid=True,
                message=f"API key is valid. Found {len(models)} available models.",
                provider="groq"
            )
        else:
            return APIKeyValidationResponse(
                valid=False,
                message="API key validation failed. Please check your key.",
                provider="groq"
            )
    except Exception as exc:
        logger.error("Groq validation error: %s", exc)
        return APIKeyValidationResponse(
            valid=False,
            message=f"Validation error: {str(exc)}",
            provider="groq"
        )


@router.post("/groq/save")
def save_groq_api_key(body: APIKeyValidationRequest):
    """Save validated Groq API key to database."""
    if not body.api_key or not body.api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="API key cannot be empty"
        )
    
    try:
        from services.llm import _check_groq
        
        # Validate first
        is_valid = _check_groq(body.api_key)
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid Groq API key"
            )
        
        # Save to database
        _save_setting("groq_api_key", body.api_key)
        
        logger.info("Groq API key saved successfully")
        
        return {
            "success": True,
            "message": "Groq API key saved successfully",
            "provider": "groq"
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error saving Groq API key: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save API key: {str(exc)}"
        )


@router.get("/groq/key-status")
def get_groq_key_status():
    """Check if Groq API key is configured."""
    cfg = get_settings()
    stored_key = _load_setting("groq_api_key") or cfg.groq_api_key
    
    return {
        "configured": bool(stored_key),
        "provider": "groq"
    }


@router.get("/models", response_model=ModelListResponse)
def get_available_models(provider: Optional[str] = None):
    """Get list of available models for a specific LLM provider or the current one."""
    cfg = get_settings()
    target_provider = provider or cfg.llm_provider
    
    try:
        if target_provider == "groq":
            from services.llm import _groq_list_models
            
            groq_key = _load_setting("groq_api_key") or cfg.groq_api_key
            if not groq_key:
                # Return configured list if no key available
                models = [m.strip() for m in cfg.groq_available_models.split(",")]
            else:
                models = _groq_list_models(groq_key)
        else:
            from services.llm import _ollama_list_models
            
            try:
                models = _ollama_list_models(cfg.ollama_base_url)
            except:
                # Fallback to configured list if Ollama is not available
                models = [m.strip() for m in cfg.ollama_available_models.split(",")]
        
        return ModelListResponse(
            provider=target_provider,
            models=models
        )
    except Exception as exc:
        logger.error("Error fetching available models for %s: %s", target_provider, exc)
        # Return fallback list
        if target_provider == "groq":
            models = [m.strip() for m in cfg.groq_available_models.split(",")]
        else:
            models = [m.strip() for m in cfg.ollama_available_models.split(",")]
        
        return ModelListResponse(
            provider=target_provider,
            models=models
        )
