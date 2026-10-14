"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


load_dotenv()


class Settings(BaseModel):
    """Non-sensitive settings used by the initial application foundation."""

    app_name: str = os.getenv("APP_NAME", "EEG Analysis API")
    app_env: str = os.getenv("APP_ENV", "development")
    app_host: str = os.getenv("APP_HOST", "127.0.0.1")
    app_port: int = int(os.getenv("APP_PORT", "8000"))

    model_checkpoint_path: Path = Path(
        os.getenv(
            "EEG_MODEL_CHECKPOINT",
            str(
                (Path(__file__).resolve().parents[3] / "models" / "cnn_bilstm_run5_best.pt")
                if (Path(__file__).resolve().parents[3] / "models" / "cnn_bilstm_run5_best.pt").exists()
                else (Path(__file__).resolve().parents[3] / "models" / "cnn_bilstm_run4_best.pt")
            ),
        )
    )

    analysis_max_upload_bytes: int = int(
        os.getenv(
            "ANALYSIS_MAX_UPLOAD_BYTES",
            str(200 * 1024 * 1024),
        )
    )

    analysis_xai_steps: int = int(
        os.getenv(
            "ANALYSIS_XAI_STEPS",
            "32",
        )
    )

    seizure_threshold: float = float(
        os.getenv(
            "EEG_SEIZURE_THRESHOLD",
            "0.50",
        )
    )

    seizure_persistence: int = int(
        os.getenv(
            "EEG_SEIZURE_PERSISTENCE",
            "1",
        )
    )


settings = Settings()
