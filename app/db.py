"""Optional PostgreSQL logging of every prediction.

If DATABASE_URL is not set (local dev, tests, Render), logging is simply disabled
and the API behaves exactly as before. A database problem never breaks /predict.
"""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger("insurance_api")


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
                                                 index=True)
    # inputs (the engineered features the model actually sees)
    age: Mapped[int] = mapped_column(Integer)
    bmi: Mapped[float] = mapped_column(Float)
    age_grp: Mapped[str] = mapped_column(String(20))
    lifestyle_risk: Mapped[str] = mapped_column(String(20))
    city_tier: Mapped[int] = mapped_column(Integer)
    income_lpa: Mapped[float] = mapped_column(Float)
    occupation: Mapped[str] = mapped_column(String(30))
    # outputs
    predicted_category: Mapped[str] = mapped_column(String(10), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(20))


_engine = None


def _normalize_url(url: str) -> str:
    """Pin the psycopg2 driver (SQLAlchemy 2.1 would otherwise pick psycopg3)."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def get_engine():
    """Create the engine lazily from DATABASE_URL; None means logging is disabled."""
    global _engine
    raw = os.getenv("DATABASE_URL")
    if not raw:
        return None
    url = _normalize_url(raw)
    if _engine is None or _engine.url.render_as_string(hide_password=False) != url:
        # Short connect timeout so an unreachable DB cannot stall predictions.
        _engine = create_engine(
            url, pool_pre_ping=True, connect_args=_connect_args(url)
        )
    return _engine


def _connect_args(url: str) -> dict:
    return {} if url.startswith("sqlite") else {"connect_timeout": 3}


def init_db() -> bool:
    try:
        engine = get_engine()
        if engine is None:
            logger.info("DATABASE_URL not set; prediction logging disabled")
            return False
        Base.metadata.create_all(engine)
        return True
    except Exception:
        logger.exception("Could not initialise database; logging disabled")
        return False


def log_prediction(user_input: dict, result: dict, model_version: str) -> None:
    """Insert one row. Swallows all errors so the prediction response is unaffected."""
    try:
        engine = get_engine()
        if engine is None:
            return
        with Session(engine) as session:
            session.add(
                PredictionLog(
                    age=user_input["age"],
                    bmi=user_input["bmi"],
                    age_grp=user_input["age_grp"],
                    lifestyle_risk=user_input["lifestyle_risk"],
                    city_tier=user_input["city_tier"],
                    income_lpa=user_input["income_lpa"],
                    occupation=user_input["occupation"],
                    predicted_category=result["predicted_category"],
                    confidence=result["confidence"],
                    model_version=model_version,
                )
            )
            session.commit()
    except Exception:
        logger.exception("Failed to log prediction")


def get_stats() -> dict | None:
    """Aggregate summary computed in SQL. None if the database is not configured."""
    engine = get_engine()
    if engine is None:
        return None
    with Session(engine) as session:
        total = session.scalar(select(func.count(PredictionLog.id))) or 0
        rows = session.execute(
            select(
                PredictionLog.predicted_category,
                func.count(PredictionLog.id),
                func.avg(PredictionLog.confidence),
            ).group_by(PredictionLog.predicted_category)
        ).all()
    return {
        "total_predictions": total,
        "by_category": {
            cat: {"count": n, "avg_confidence": round(float(avg), 4)}
            for cat, n, avg in rows
        },
    }
