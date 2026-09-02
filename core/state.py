from pydantic import BaseModel, Field

from models.hypothesis import (
    Hypothesis,
    HypothesisReview,
    Paper,
    SimilarityPair,
    ResearchRoadmap,
)


class ScientistState(BaseModel):
    """Shared state maintained by the supervisor."""

    research_goal: str

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    reviews: list[HypothesisReview] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)
    similarities: list[SimilarityPair] = Field(
        default_factory=list
    )
    roadmap: ResearchRoadmap | None = None

    current_generation: int = 0