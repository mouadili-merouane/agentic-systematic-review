"""Agent Extraction & Évaluation : extraction structurée + risque de biais (PROBAST simplifié).

Les données sont extraites uniquement des passages retrouvés par le RAG dans l'article ; chaque champ
clé est accompagné d'une citation exacte (evidence) que l'agent critique vérifie ensuite dans le texte.
"""
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, BeforeValidator, Field
from ..llm import call_json


def _rating(v):
    v = str(v or "").strip().lower()
    return v if v in ("low", "high") else "unclear"


def _validation(v):
    v = str(v or "").strip().lower()
    ext, inte = "external" in v, any(k in v for k in ("internal", "cross", "split", "bootstrap", "hold"))
    if ext and inte:
        return "internal+external"
    if ext:
        return "external"
    if inte:
        return "internal"
    return "none" if v in ("none", "no", "not performed") else "unclear"


Rating = Annotated[Literal["low", "high", "unclear"], BeforeValidator(_rating)]
Validation = Annotated[Literal["internal", "external", "internal+external", "none", "unclear"],
                       BeforeValidator(_validation)]


class Evidence(BaseModel):
    field: str
    quote: str = Field(description="verbatim sentence fragment copied from the passages (max 30 words)")


class Domain(BaseModel):
    rating: Rating
    rationale: str


class RiskOfBias(BaseModel):
    participants: Domain
    predictors: Domain
    outcome: Domain
    analysis: Domain
    overall: Rating
    applicability_concern: Rating


class Extraction(BaseModel):
    country: str = "not reported"
    data_source: str = Field("not reported", description="registry, EHR, trial, hospital name...")
    study_design: str = "not reported"
    sample_size: Optional[int] = Field(None, description="number of patients used for modelling")
    hf_population: str = Field("not reported", description="HF type / setting (HFrEF, acute, post-discharge...)")
    outcome: str = Field("not reported", description="e.g. 30-day readmission, 1-year mortality")
    ml_models: list[str] = Field(default_factory=list)
    best_model: str = "not reported"
    best_auc: Optional[float] = Field(None, description="AUC / C-statistic of the best ML model (0-1)")
    best_auc_ci: str = Field("not reported", description="95% CI as text, e.g. 0.74-0.80")
    comparator: str = Field("none", description="conventional model/score compared, if any")
    comparator_auc: Optional[float] = None
    validation: Validation = "unclear"
    calibration_reported: Optional[bool] = None
    explainability: str = Field("none", description="SHAP, feature importance... or none")
    key_finding: str = Field(description="one sentence")
    evidence: list[Evidence] = Field(description="quotes supporting sample_size, outcome, best_auc, validation")
    risk_of_bias: RiskOfBias


SYSTEM = """You are the data-extraction and appraisal agent of a systematic review of prediction-model studies.
Extract the fields ONLY from the supplied passages; write 'not reported' (or null) when absent — never guess.
Numbers must be copied exactly as written. For evidence quotes, copy text verbatim from the passages.
Assess risk of bias with PROBAST domains (participants, predictors, outcome, analysis):
'high' if a signalling concern is clearly present (e.g. <100 events, no validation, no handling of missing data,
predictors measured after outcome), 'low' if clearly addressed, otherwise 'unclear'.
Research question: {question}"""

QUERIES = [
    "study population data source cohort country inclusion criteria number of patients",
    "machine learning models algorithms random forest gradient boosting neural network logistic regression",
    "primary outcome readmission mortality follow-up prediction horizon",
    "results AUC C-statistic area under the curve 95% confidence interval performance comparison",
    "validation cross-validation external validation calibration missing data imputation",
]


def run(llm, question: str, record: dict, passages: list[str]) -> Extraction:
    ev = "\n\n".join(f"[passage {i+1}] {t}" for i, t in enumerate(passages))
    user = (f"TITLE: {record['title']} ({record.get('year')})\nABSTRACT: {record.get('abstract','')[:2500]}"
            f"\n\nPASSAGES FROM FULL TEXT:\n{ev}")
    return call_json(llm, SYSTEM.format(question=question), user, Extraction)
