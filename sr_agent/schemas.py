"""Schémas de données échangés entre agents (validés par Pydantic)."""
from typing import Optional
from pydantic import BaseModel, Field


class Criterion(BaseModel):
    id: str = Field(description="I1, I2... for inclusion; E1, E2... for exclusion")
    text: str = Field(description="Operational, checkable criterion")


class Protocol(BaseModel):
    title: str
    research_question: str
    pico: dict[str, str] = Field(description="keys: population, intervention, comparator, outcomes")
    inclusion_criteria: list[Criterion]
    exclusion_criteria: list[Criterion]
    study_designs: list[str]
    years: list[int]
    languages: list[str]
    rationale: str = Field(description="Short justification of the operationalised criteria")


class ConceptBlock(BaseModel):
    name: str = Field(description="e.g. population, intervention, outcome")
    keywords: list[str] = Field(description="free-text synonyms / spelling variants (5-12)")
    mesh_terms: list[str] = Field(default_factory=list, description="exact MeSH headings, if any")


class SearchStrategy(BaseModel):
    concept_blocks: list[ConceptBlock] = Field(description="2-4 blocks combined with AND; synonyms combined with OR")
    exclusion_keywords: list[str] = Field(default_factory=list, description="terms to NOT (e.g. pediatric)")
    rationale: str


class Record(BaseModel):
    id: str
    title: str
    abstract: str = ""
    year: Optional[int] = None
    authors: list[str] = []
    venue: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    source: str = ""
    found_in: list[str] = []
