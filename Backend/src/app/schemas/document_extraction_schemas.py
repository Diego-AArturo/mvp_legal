# app/schemas/document_extraction_schemas.py
"""
Schemas for document information extraction (people, IDs, roles).

This module defines Pydantic models for extracting structured information
from legal documents, particularly for tutela cases, using Ollama-backed flows.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PersonInfo(BaseModel):
    """
    Information about a person mentioned in a legal document.

    Contains name, role, and identification number (cédula).
    """

    nombre_completo: str = Field(
        ...,
        description="Full name of the person (e.g., 'Juan Pérez García')",
        min_length=1,
    )

    rol: str = Field(
        ...,
        description="Role of the person in the document (e.g., 'accionante de la tutela', 'abogada responsable')",
        min_length=1,
    )

    cedula: str = Field(
        ...,
        description="Colombian ID number (cédula de ciudadanía), digits only",
        pattern=r"^\d{6,11}$",
    )

    tipo_documento: Optional[str] = Field(
        default="CC",
        description="Document type (CC, CE, TI, etc.)",
    )

    contexto: Optional[str] = Field(
        default=None,
        description="Additional context about the person from the document",
    )

    @field_validator("cedula")
    @classmethod
    def validate_cedula(cls, v: str) -> str:
        """Validate Colombian ID format."""
        # Remove any non-digit characters
        v = "".join(c for c in v if c.isdigit())

        if len(v) < 6 or len(v) > 11:
            raise ValueError("Cédula must be between 6 and 11 digits")

        return v

    @field_validator("nombre_completo")
    @classmethod
    def validate_nombre(cls, v: str) -> str:
        """Normalize name formatting."""
        # Remove extra whitespace and capitalize properly
        v = " ".join(v.split())
        return v.strip()


class DocumentPeopleExtraction(BaseModel):
    """
    Complete extraction of people information from a legal document.

    Contains a list of all people identified in the document with their
    roles and identification numbers.
    """

    personas: List[PersonInfo] = Field(
        default_factory=list,
        description="List of people identified in the document",
    )


class PersonExtractionRequest(BaseModel):
    """
    Request for extracting people information from a document.
    """

    texto: str = Field(
        ...,
        description="Document text to extract people information from",
        min_length=10,
    )

    tipo_documento: Optional[str] = Field(
        default="tutela",
        description="Type of document being processed",
    )

    max_tokens: Optional[int] = Field(
        default=2000,
        description="Maximum tokens for generation",
        ge=100,
        le=4000,
    )

    temperature: Optional[float] = Field(
        default=0.1,
        description="Temperature for generation (lower = more deterministic)",
        ge=0.0,
        le=1.0,
    )

    incluir_contexto: bool = Field(
        default=True,
        description="Include contextual information about each person",
    )


class PersonExtractionResponse(BaseModel):
    """
    Response from person extraction service.
    """

    success: bool = Field(
        default=True,
        description="Whether extraction was successful",
    )

    personas: List[PersonInfo] = Field(
        default_factory=list,
        description="List of people extracted from the document",
    )

    total_personas: int = Field(
        default=0,
        description="Total number of people found",
    )

    resumen: Optional[str] = Field(
        default=None,
        description="Summary of the extraction",
    )

    raw_response: Optional[str] = Field(
        default=None,
        description="Raw JSON response from the model",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if extraction failed",
    )


class CedulaValidationResult(BaseModel):
    """Result of cédula validation."""

    cedula: str = Field(..., description="Validated cédula")
    is_valid: bool = Field(..., description="Whether the cédula is valid")
    formatted: str = Field(..., description="Formatted cédula (with separators)")
    error: Optional[str] = Field(default=None, description="Validation error if any")
