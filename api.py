from __future__ import annotations

import hmac
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from celery import Celery

load_dotenv(".env")


SECRET_KEY: str = os.getenv("SECRET_KEY", "")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

REDIS_BROKER: str = os.getenv("REDIS_BROKER", "redis://redis:6379/0")
REDIS_BACKEND: str = os.getenv("REDIS_BACKEND", "redis://redis:6379/1")

INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/photos"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    measurements = relationship(
        "MeasurementModel", back_populates="user", cascade="all, delete-orphan"
    )


class MeasurementModel(Base):
    __tablename__ = "measurements"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    chest_cm = Column(Float, nullable=True)
    waist_cm = Column(Float, nullable=True)
    hips_cm = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    body_type = Column(String, nullable=True)
    source = Column(String, default="manual", nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    user = relationship("UserModel", back_populates="measurements")


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


celery_client = Celery(
    "coty_body_api",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND,
)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserModel:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Неверный или просроченный токен",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise exc
    except JWTError:
        raise exc

    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if user is None:
        raise exc
    return user


def require_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    """Только для запросов от воркера."""
    if not INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY не настроен")
    if not x_internal_key or not hmac.compare_digest(x_internal_key, INTERNAL_API_KEY):
        raise HTTPException(status_code=403, detail="Нет доступа")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


class MeasurementOut(BaseModel):
    id: str
    chest_cm: Optional[float]
    waist_cm: Optional[float]
    hips_cm: Optional[float]
    height_cm: Optional[float]
    body_type: Optional[str]
    source: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MeasurementCreate(BaseModel):
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hips_cm: Optional[float] = None
    height_cm: Optional[float] = None
    body_type: Optional[str] = None
    notes: Optional[str] = None


class InternalMeasurementCreate(BaseModel):
    user_id: str
    chest_cm: Optional[float] = None
    waist_cm: Optional[float] = None
    hips_cm: Optional[float] = None
    height_cm: Optional[float] = None
    body_type: Optional[str] = None
    source: str = "auto"


class AnalyzeResponse(BaseModel):
    task_id: str
    message: str = "Фото поставлено в очередь на обработку"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None


def _validate_image(file: UploadFile, content: bytes) -> None:
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Файл слишком большой. Максимум 10 МБ.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Разрешены форматы: jpeg, png, webp.")


def _save_tmp(content: bytes, original_filename: str) -> Path:
    suffix = Path(original_filename).suffix or ".jpg"
    path = TMP_DIR / f"{uuid.uuid4()}{suffix}"
    path.write_bytes(content)
    return path

app = FastAPI(
    title="coty_body API",
    version="2.0.0",
    description="API для определения типа фигуры женщины по фото анфас и профиль.",
)


@app.post("/auth/register", response_model=TokenResponse, status_code=201, tags=["auth"])
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.query(UserModel).filter(UserModel.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email уже зарегистрирован")
    user = UserModel(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    user = db.query(UserModel).filter(UserModel.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")
    return TokenResponse(access_token=create_access_token(user.id))


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def get_me(current_user: UserModel = Depends(get_current_user)) -> UserOut:
    return current_user


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    front_image: UploadFile = File(..., description="Фото анфас"),
    side_image: UploadFile = File(..., description="Фото профиль"),
    height_cm: float = Form(..., description="Рост в сантиметрах"),
    current_user: UserModel = Depends(get_current_user),
) -> AnalyzeResponse:
    front_bytes = await front_image.read()
    side_bytes = await side_image.read()

    _validate_image(front_image, front_bytes)
    _validate_image(side_image, side_bytes)

    front_path = _save_tmp(front_bytes, front_image.filename or "front.jpg")
    side_path = _save_tmp(side_bytes, side_image.filename or "side.jpg")

    task = celery_client.send_task(
        "analyze_photo",
        args=[str(front_path), str(side_path), height_cm, current_user.id],
    )
    return AnalyzeResponse(task_id=task.id)


@app.get("/analyze/{task_id}", response_model=TaskStatusResponse, tags=["analyze"])
def get_task_status(
    task_id: str,
    current_user: UserModel = Depends(get_current_user),
) -> TaskStatusResponse:
    async_result = celery_client.AsyncResult(task_id)
    return TaskStatusResponse(
        task_id=task_id,
        status=async_result.status,
        result=async_result.result if async_result.status == "SUCCESS" else None,
    )


@app.get("/me/measurements", response_model=List[MeasurementOut], tags=["measurements"])
def list_measurements(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[MeasurementOut]:
    return (
        db.query(MeasurementModel)
        .filter(MeasurementModel.user_id == current_user.id)
        .order_by(MeasurementModel.created_at.desc())
        .all()
    )


@app.get("/me/measurements/{measurement_id}", response_model=MeasurementOut, tags=["measurements"])
def get_measurement(
    measurement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MeasurementOut:
    m = (
        db.query(MeasurementModel)
        .filter(
            MeasurementModel.id == measurement_id,
            MeasurementModel.user_id == current_user.id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    return m


@app.post("/me/measurements", response_model=MeasurementOut, status_code=201, tags=["measurements"])
def create_measurement(
    body: MeasurementCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MeasurementOut:
    m = MeasurementModel(
        user_id=current_user.id,
        source="manual",
        **body.model_dump(exclude_none=True),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


@app.put("/me/measurements/{measurement_id}", response_model=MeasurementOut, tags=["measurements"])
def update_measurement(
    measurement_id: str,
    body: MeasurementCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MeasurementOut:
    m = (
        db.query(MeasurementModel)
        .filter(
            MeasurementModel.id == measurement_id,
            MeasurementModel.user_id == current_user.id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(m, field, value)
    m.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(m)
    return m


@app.delete("/me/measurements/{measurement_id}", status_code=204, tags=["measurements"])
def delete_measurement(
    measurement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    m = (
        db.query(MeasurementModel)
        .filter(
            MeasurementModel.id == measurement_id,
            MeasurementModel.user_id == current_user.id,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    db.delete(m)
    db.commit()

@app.post(
    "/internal/measurements",
    response_model=MeasurementOut,
    status_code=201,
    tags=["internal"],
    include_in_schema=False,
)
def internal_create_measurement(
    body: InternalMeasurementCreate,
    _: None = Depends(require_internal_key),
    db: Session = Depends(get_db),
) -> MeasurementOut:
    m = MeasurementModel(**body.model_dump(exclude_none=True))
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

@app.get("/health", tags=["system"])
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
