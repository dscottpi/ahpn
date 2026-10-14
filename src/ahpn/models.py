from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    ahpn_id: Mapped[int] = mapped_column(unique=True, index=True)
    source_page: Mapped[str] = mapped_column(String(500))
    visibility: Mapped[str] = mapped_column(String(16))
    page_count: Mapped[int | None]
    document_type: Mapped[str | None] = mapped_column(String(32))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    pages: Mapped[list[Page]] = relationship(back_populates="document", cascade="all, delete-orphan")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_number: Mapped[int]
    image_key: Mapped[str] = mapped_column(String(500))
    content_type: Mapped[str] = mapped_column(String(100))
    byte_count: Mapped[int]

    document: Mapped[Document] = relationship(back_populates="pages")
    lines: Mapped[list[OcrLine]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="OcrLine.line_index",
    )


class OcrLine(Base):
    """One Textract LINE block. Kept so chunks can be rebuilt without another OCR call."""

    __tablename__ = "ocr_lines"
    __table_args__ = (UniqueConstraint("page_id", "line_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    line_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    top: Mapped[float] = mapped_column(Float)
    left: Mapped[float] = mapped_column(Float)
    width: Mapped[float] = mapped_column(Float)
    height: Mapped[float] = mapped_column(Float)

    page: Mapped[Page] = relationship(back_populates="lines")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int]
    page_end: Mapped[int]
    char_count: Mapped[int]
    confidence: Mapped[float] = mapped_column(Float)
    line_refs: Mapped[list] = mapped_column(JSON)

    document: Mapped[Document] = relationship(back_populates="chunks")
