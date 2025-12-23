# modelsv1.py - Shared data models
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    speaker: str
    text: str

class OutlineItem(BaseModel):
    id: str
    question: str
    importance: str
    keywords: List[str] = []

class OutlineSection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str
    section: str
    questions: List[OutlineItem]

class Persona(BaseModel):
    model_config = ConfigDict(extra='allow')
    name: str
    age: int
    personality: str

class DivergentAnalysis(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # Existing fields
    emotional_insights: List[str] = Field(default_factory=list, description="Key emotional aspects in the response")
    chain_of_thought: List[str] = Field(default_factory=list, description="Chain of concepts in the response")
    uncovered_angles: List[str] = Field(default_factory=list, description="Potential exploration angles")
    followup_questions: List[str] = Field(default_factory=list, description="Generated follow-up questions, in format 'Type: Question'")
    memory_snippets: List[str] = Field(default_factory=list, description="Key information worth saving into long-term memory")
    
    # NEW: Outline matching results
    matched_question_id: Optional[str] = Field(None, description="ID of the best matching outline question")
    matched_section_id: Optional[str] = Field(None, description="ID of the section containing the matched question")
    match_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in the outline match")
    coverage_assessment: float = Field(0.0, ge=0.0, le=1.0, description="How well response covers the matched question")

class QuestionScore(BaseModel):
    model_config = ConfigDict(extra='forbid')
    score: float = Field(..., ge=0.0, le=1.0, description="Base score for the question")
    novelty_score: float = Field(..., ge=0.0, le=1.0, description="Novelty score for the question")
    depth_score: int = Field(..., ge=1, le=3, description="Depth level: 1=surface, 2=detailed, 3=deep")
    reason: str = Field(..., description="Reason for the score")

class CoverageOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    next_question: str = Field(
        ..., 
        description="The next question to ask"
    )
    reasoning: str = Field(
        ..., 
        description="Strategy reasoning for question selection"
    )
    exploration_strategy: str = Field(
        ..., 
        description="Strategy used: within-section/cross-section/priority-jump"
    )
    novelty_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Novelty score for the question"
    )
    depth_score: int = Field(
        ..., 
        ge=1, 
        le=3,
        description="Depth level of the question: 1=surface, 2=detailed, 3=deep"
    )

class FollowupOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    followup_question: str
    reasoning: str
    followup_type: str = Field(
        ..., description="Type of follow-up: 'deep-dive', 'clarification', 'emotional', or 'exploratory'"
    )
    priority: int = Field(
        1, description="Priority ranking (1=highest) for follow-up questions"
    )
    depth: int = Field(1, description="Depth level of the follow-up question")

def flatten_outline_sections(sections: List[OutlineSection]) -> List[OutlineItem]:
    all_questions = []
    for section in sections:
        all_questions.extend(section.questions)
    return all_questions

def load_outline_from_sections(file_path: str) -> List[OutlineSection]:
    import json
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return [OutlineSection(**section) for section in data]

class CoverageOutputRand(BaseModel):
    """Output model for the randomized convergent agent"""
    question_scores: List[QuestionScore] = Field(
        ..., 
        description="List of scores and reasons for each question"
    )
    real_questions: List[str] = Field(
        ..., 
        description="List of questions that can be asked next"
    )
    question_types: List[str] = Field(
        ..., 
        description="List of question types (broad/depth/balanced) for each question"
    )
    reasoning: str = Field(
        ..., 
        description="Strategy reasoning for question selection"
    )
    exploration_strategy: str = Field(
        ..., 
        description="Strategy used: within-section/cross-section/priority-jump"
    )