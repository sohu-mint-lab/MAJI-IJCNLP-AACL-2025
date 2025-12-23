from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict
from agents import Agent, Runner,set_default_openai_api,set_default_openai_client,set_tracing_disabled
from sentence_transformers import SentenceTransformer
from .rag_memory import RAGMemorySystem
import os, warnings, json, random, math
from openai import AsyncOpenAI
from .modelsv1 import (
    ConversationTurn, OutlineSection, OutlineItem, Persona,
    DivergentAnalysis, CoverageOutput, FollowupOutput,
    flatten_outline_sections, CoverageOutputRand
)
# Import V2's ConvergentAgent and models
from ..V2.modelsv2 import DivergentQuestion, ConvergentAgentOutput
from ..V2.DCAgentV2 import ConvergentAgent
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# DC AGENTS with better outline exploration focus
DAgent = Agent(
    name="Divergent Thinking Agent",
    instructions=(
        "You are the thinking engine of an interview, responsible for divergent thinking analysis.\n"
        
        "Core Tasks:\n"
        "1. **Outline Matching**: Analyze the matching degree between the current answer and the questions in the outline\n"
        "2. Emotional Insight: Identify emotional expressions in the answer\n"
        "3. Thinking Chain: Analyze logical relationships and conceptual connections\n"
        "4. Unexplored Perspectives: Identify important areas in the outline that are not fully covered\n"
        "5. Follow-up Questions: Generate exploratory questions that can be extended from the current conversation\n"
        "6. Memory Snippets: Extract key information\n\n"
        
        "**Outline Matching Requirements**:\n"
        "- Carefully compare the relationship between the answer content and each outline question\n"
        "- Pay special attention to the connection between the answer and the previous question\n"
        "- If the answer is an in-depth exploration of the previous question, it should be regarded as a deep answer to that question\n"
        "- matched_question_id: The best matching question ID, composed of S + number + Q + number (e.g., 'S1Q2') or null\n"
        "- matched_section_id: Corresponding section ID, composed of S + number (e.g., 'S1') or null\n"
        "- match_confidence: Matching confidence (0.0-1.0)\n"
        "- coverage_assessment: Coverage level of the answer to the question (0.0-1.0)\n\n"
        
        "**Matching Judgment Criteria**:\n"
        "- Directly answers outline question: confidence 0.8+, coverage 0.7+\n"
        "- Partially related: confidence 0.5-0.8, coverage 0.3-0.7\n"
        "- Indirectly mentioned: confidence 0.3-0.5, coverage 0.1-0.3\n"
        "- Unrelated: confidence 0.0, coverage 0.0\n\n"
        
        "**Output Requirements**:\n"
        "- Must output valid JSON format\n"
        "- Include all DivergentAnalysis fields (including new matching fields)\n"
        "- Do not include any text outside of JSON\n\n"

        "**Follow-up Question Generation Requirements**:\n"
        "1. Based on the current conversation content, generate at least three next questions suitable for continuing the interview\n"
        "2. Question types should cover breadth, depth, and comprehensiveness\n"
        "3. Each question must be labeled with a type (broad/depth/balanced)\n"
        "4. Provide strategic reasons for question generation\n"
        "5. Ensure questions are natural, fluent, and fit the interview context. Do not use the original text from the outline directly. Do not be too brief. Follow the logic and tone of an interview conversation.\n"
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAnalysis,
)
 
        # "Decision Strategy (by priority):\n"
        # "0. **Breadth-first**: Ask three questions in the same section consecutively, then jump to other related sections"
        # "1. **Depth-first**: If current matched question coverage < 0.7, continue in-depth exploration.\n"
        # "2. **Section Completed**: If current section coverage > 70%, jump to new section\n"
        # "3. **High Priority First**: Prioritize exploring critical/high importance uncovered questions\n"
        # "4. **Balanced Exploration**: Avoid excessive neglect of any section\n\n"
        
CAgent = Agent(
    name="Convergent Thinking Agent",
    instructions=(
        "You are the strategy director of the interview, generating the next question based on the full divergent analysis results.\n"
       
        "**exploration_strategy types**:\n"
        "- 'within-section': Deepen within the current section\n"
        "- 'cross-section': Jump to another section\n"
        "- 'priority-jump': Jump to high priority uncovered questions\n"
        "- 'balance-exploration': Balanced exploration\n\n"
        
        "**Key Output Constraints**:\n"
        "- next_question: The next question to ask, must be natural, fluent, and fit the interview context\n"
        "- reasoning: Detailed reasons for strategy selection\n"
        "- exploration_strategy: Must be one of the four types above\n"
        "- novelty_score: Decimal 0.0-1.0, representing the novelty of the question\n"
        "- depth_score: Integer 1-3, representing the depth of the question\n\n"
        
        "**Depth Scoring Criteria**:\n"
        "- 1: Surface level question, obtaining basic information\n"
        "- 2: In-depth question, exploring details and reasons\n"
        "- 3: Deep level question, exploring essence and impact\n\n"
        
        "**Novelty Scoring Criteria**:\n"
        "- 0.0-0.3: Repetitive or similar to existing questions\n"
        "- 0.3-0.6: Partial new perspective\n"
        "- 0.6-0.8: New topic or new perspective\n"
        "- 0.8-1.0: Completely innovative question\n\n"
        
        "Strict Requirements:\n"
        "1. All numerical values must be within a reasonable range\n"
        "2. Questions must be natural, fluent, and fit the interview context\n"
        "3. Must strictly follow JSON format, do not include any extra text\n"
        "4. Do not directly use the original text from the outline\n"
    ),
    model="gpt-4.1-mini",
    output_type=CoverageOutput,
)

CAgentRand = Agent(
    name="Randomized Convergent Thinking Agent",
    instructions=(
        "You are the randomized strategy director of the interview, making probabilistic decisions based on the full divergent analysis results.\n"
        
        "Core Tasks:\n"
        "1. Generate multi-dimensional scores for each possible subsequent question\n"
        "2. Generate a real piece of question for each potential question\n"
        "3. Label each question with a type (broad/depth/balanced)\n"
        "4. Provide strategy reasons and exploration directions\n\n"
        
        "**Question Type Definitions**:\n"
        "- broad: Breadth-first questions, exploring new topics or angles\n"
        "- depth: Depth-first questions, exploring current topics in-depth\n"
        "- balanced: Balanced questions, considering both breadth and depth\n\n"
        
        "**Scoring Criteria**:\n"
        "1. Base score (score): 0.0-1.0\n"
        "   - Based on question importance and coverage\n"
        "   - Consider coherence with the current conversation\n"
        "   - Evaluate natural fluency of the question\n\n"
        
        "2. Novelty score (novelty_score): 0.0-1.0\n"
        "   - 0.0-0.3: Repetitive or similar to existing questions\n"
        "   - 0.3-0.6: Partial new perspective\n"
        "   - 0.6-0.8: New topic or new perspective\n"
        "   - 0.8-1.0: Completely innovative question\n\n"
        
        "3. Depth score (depth_score): 1-3\n"
        "   - 1: Surface level question, obtaining basic information\n"
        "   - 2: In-depth question, exploring details and reasons\n"
        "   - 3: Deep level question, exploring essence and impact\n\n"
        
        "**Output Format Requirements**:\n"
        "Must be strictly in the following JSON format:\n"
        "{\n"
        '  "question_scores": [\n'
        '    {"score": 0.8, "novelty_score": 0.7, "depth_score": 2, "reason": "reason 1"},\n'
        '    {"score": 0.6, "novelty_score": 0.5, "depth_score": 1, "reason": "reason 2"}\n'
        "  ],\n"
        '  "real_questions": ["question 1", "question 2"],\n'
        '  "question_types": ["broad", "depth"],\n'
        '  "reasoning": "strategy selection reason",\n'
        '  "exploration_strategy": "within-section"\n'
        "}\n\n"
        
        "**Field Descriptions**:\n"
        "- question_scores: List of multi-dimensional scores for each question\n"
        "- real_questions: Corresponding real-world question list, length must match question_scores\n"
        "- question_types: List of types for each question, must be broad/depth/balanced\n"
        "- reasoning: Strategy selection reasons\n"
        "- exploration_strategy: Must be one of within-section/cross-section/priority-jump\n\n"
        
        "Strict Requirements:\n"
        "1. All numerical values must be within a reasonable range\n"
        "2. All list lengths must be identical\n"
        "3. All text must be in English\n"
        "4. Questions must be natural, fluent and fit the interview context\n"
        "5. Must strictly follow JSON format, do not include any extra text"
    ),
    model="gpt-4.1-mini",
    output_type=CoverageOutputRand
)

def calculate_softmax_probabilities(
    convergent_scores: List[float], 
    user_preference_scores: List[float],
    novelty_scores: List[float],
    depth_scores: List[int],
    temperature: float = 1.0,
    lambda_preference: float = 0.5
) -> List[float]:
    """Calculate softmax probabilities from combined scores"""
    if not convergent_scores or len(convergent_scores) != len(user_preference_scores) or \
       len(convergent_scores) != len(novelty_scores) or len(convergent_scores) != len(depth_scores):
        return []
    
    # Normalize depth scores to 0-1 range
    normalized_depth_scores = [(d - 1) / 2 for d in depth_scores]  # Convert 1-3 to 0-1
    
    # Combine base score with novelty and depth (as part of the base score)
    enhanced_scores = [
        s_i * (1 + 0.3 * n_i + 0.2 * d_i)  # Novelty and depth enhance the base score
        for s_i, n_i, d_i in zip(convergent_scores, novelty_scores, normalized_depth_scores)
    ]
    
    # Apply user preference with lambda
    combined_scores = [
        (s_i + lambda_preference * u_i) / temperature 
        for s_i, u_i in zip(enhanced_scores, user_preference_scores)
    ]
    
    # Apply softmax to combined scores
    max_score = max(combined_scores)
    exp_scores = [math.exp(s - max_score) for s in combined_scores]
    sum_exp = sum(exp_scores)
    probabilities = [exp_score / sum_exp for exp_score in exp_scores]
    
    return probabilities

def select_question_by_probability(questions: List[str], probabilities: List[float]) -> str:
    """Select a question randomly based on probability distribution"""
    if not questions or not probabilities or len(questions) != len(probabilities):
        raise ValueError("Questions and probabilities must be non-empty and of equal length")
    
    # Generate random number between 0 and 1
    r = random.random()
    print(f"Random number for selection: {r:.4f}")
    # Find the selected question based on cumulative probability
    cumulative_prob = 0
    for question, prob in zip(questions, probabilities):
        cumulative_prob += prob
        if r <= cumulative_prob:
            return question
    
    # Fallback to last question (should not happen due to probability normalization)
    return questions[-1]

class DCAgentSystem:
    def __init__(
        self,
        sections: List[OutlineSection],
        persona: Persona,
        session_id: str | None = None,
        use_rag: bool = True,
        use_random_agent: bool = False,
        user_preference: str = "balanced",
        temperature: float = 1.0,
        lambda_preference: float = 0.5,
    ):
        self.sections = sections
        # Add deep dive section
        self.deep_dive_section = OutlineSection(
            id="D",
            section="Deep Dives",
            questions=[]
        )
        self.sections.append(self.deep_dive_section)
        
        self.flattened_questions = flatten_outline_sections(sections)
        self.persona = persona
        self.session_id = session_id 
        self.use_rag = use_rag
        self.use_random_agent = use_random_agent
        self.user_preference = user_preference
        self.temperature = temperature
        self.lambda_preference = lambda_preference

        # Update section_coverage to store both coverage and depth scores
        self.section_coverage: Dict[str, Dict[str, List[Dict[str, float]]]] = {
            s.id: {q.id: [] for q in s.questions} for s in sections
        }

        if use_rag:
            self.rag_memory = RAGMemorySystem()

        self.covered_keywords, self.uncovered_aspects = set(), []
        self.emotional_tracker, self.chain_tracker = [], []
        self.followups: List[FollowupOutput] = []
        self.asked_question_texts: set[str] = set()  # Track question text, not IDs
        self.last_coverage: Optional[CoverageOutput] = None
        self.last_coverage_rand: Optional[CoverageOutputRand] = None
    
    def get_section_summary(self):
        summ = {}
        for sec in self.sections:
            all_scores = [
                s["coverage"] for q in sec.questions 
                for s in self.section_coverage[sec.id][q.id]
            ]
            covered = sum(bool(self.section_coverage[sec.id][q.id]) for q in sec.questions)
            avg_depth = sum(
                s["depth"] for q in sec.questions 
                for s in self.section_coverage[sec.id][q.id]
            ) / len(all_scores) if all_scores else 0.0
            
            summ[sec.id] = {
                "section_name": sec.section,
                "avg_coverage": round(sum(all_scores) / len(all_scores), 2)
                if all_scores else 0.0,
                "avg_depth": round(avg_depth, 2) if all_scores else 0.0,
                "covered_questions": covered,
                "total_questions": len(sec.questions),
                "coverage_percentage": round(covered / len(sec.questions) * 100, 1)
                if sec.questions else 0,
            }
        return summ

    def get_section_coverage_percentage(self, section_id: str) -> float:
        """Calculate coverage percentage for a section"""
        if section_id not in self.section_coverage:
            return 0.0
        
        section = next(s for s in self.sections if s.id == section_id)
        covered_count = sum(
            1 for q in section.questions 
            if self.section_coverage[section_id][q.id]
        )
        return (covered_count / len(section.questions)) * 100 if section.questions else 0

    def get_high_priority_uncovered_questions(self) -> List[Tuple[OutlineSection, OutlineItem]]:
        """Get uncovered questions sorted by importance"""
        uncovered = []
        importance_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        for sec in self.sections:
            for q in sec.questions:
                if not self.section_coverage[sec.id][q.id]:
                    uncovered.append((sec, q))
        
        # Sort by importance only
        uncovered.sort(key=lambda x: importance_order.get(x[1].importance.lower(), 4))
        return uncovered

    def format_outline_with_coverage(self) -> str:
        """Format outline showing coverage statistics"""
        lines = []
        for sec in self.sections:
            coverage_pct = self.get_section_coverage_percentage(sec.id)
            lines.append(f"\n=== {sec.section} (Section {sec.id}) - Coverage: {coverage_pct:.1f}% ===")
            
            for q in sec.questions:
                scores = self.section_coverage[sec.id][q.id]
                if not scores:
                    mark = "X"
                else:
                    avg = sum(s["coverage"] for s in scores) / len(scores)
                    mark = "V" if avg > 0.6 else "~"
                
                lines.append(f"[{mark}] {q.id} ({q.importance}, depth {q.depth}): {q.question}")
        return "\n".join(lines)

    def format_persona(self) -> str:
        p_dict = self.persona.model_dump()
        lines = ["--- INTERVIEWEE PERSONA ---"]
        for key, value in p_dict.items():
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)
        
    def _get_rag_context(self) -> str:
        """Return concise RAG info for CAgent prompt."""
        if not self.use_rag:
            return "RAG disabled."
        
        pending = self.rag_memory.get_pending_followups(self.session_id)
        topic_cov = self.rag_memory.get_topic_coverage_summary(self.session_id)
        parts: List[str] = []
        
        if pending:
            parts.append("PENDING FOLLOW-UPS (top 3):")
            for pf in pending[:3]:
                parts.append(f"- R{pf['round']} {pf['description']}")
        
        if topic_cov:
            low = sorted(topic_cov.items(), key=lambda x: x[1]["coverage"])[:3]
            parts.append("LOW-COVERAGE TOPICS:")
            for tid, info in low:
                qtxt = info.get("question", info.get("prompt", ""))
                parts.append(f"- {tid} ({info['coverage']:.2f}) {qtxt}")
        
        memory = self.rag_memory.get_long_term_memory(self.session_id, limit=3)
        if memory:
            parts.append("RECENT MEMORY:")
            for s in memory:
                parts.append(f"- {s[:120]}")

        return "\n".join(parts) or "RAG has nothing yet."

    def _create_outline_context_for_divergent(self) -> str:
        """Create formatted outline context for divergent agent"""
        lines = ["=== OUTLINE FOR MATCHING ==="]
        
        for sec in self.sections:
            coverage_pct = self.get_section_coverage_percentage(sec.id)
            lines.append(f"\n[{sec.id}] {sec.section} - Coverage: {coverage_pct:.1f}%")
            
            for q in sec.questions:
                scores = self.section_coverage[sec.id][q.id]
                status = "V" if scores and (sum(s["coverage"] for s in scores)/len(scores)) > 0.6 else "X"
                lines.append(f"  {status} {q.id}: {q.question} ({q.importance})")
        
        return "\n".join(lines)

    async def process_response(self, response: str, round_num: int | None = None, transcript: List[ConversationTurn] | None = None) -> Tuple[DivergentAnalysis, CoverageOutput | CoverageOutputRand]:
        
        print("\n==== DIVERGENT THINKING WITH OUTLINE MATCHING ====")
        
        # Get the previous question if available, otherwise use first outline question
        previous_question = None
        if transcript and len(transcript) >= 2:
            previous_question = transcript[-2].text
        else:
            # Use first question from outline if no transcript
            previous_question = self.sections[0].questions[0].question
        
        # Enhanced context for divergent agent with outline and previous question
        d_ctx = (
            f"PERSONA:\n{self.format_persona()}\n\n"
            f"PREVIOUS QUESTION:\n{previous_question}\n\n"
            f"CURRENT RESPONSE TO ANALYZE:\n{response}\n\n"
            f"{self._create_outline_context_for_divergent()}\n\n"
            f"PREVIOUS COVERAGE SUMMARY:\n{json.dumps(self.get_section_summary(), ensure_ascii=False, indent=2)}\n\n"
            f"RAG CONTEXT:\n{self._get_rag_context()}"
        )
        
        print(f"DAgent context: {len(d_ctx)} chars")
        d_res = await Runner.run(DAgent, d_ctx)
        danal = d_res.final_output_as(DivergentAnalysis)
        
        # Update section coverage based on divergent analysis
        is_deep_dive = False
        if danal.coverage_assessment > 0:
            if danal.matched_question_id and danal.matched_section_id:
                # Direct match - update normal coverage
                depth_level = 1  # Default depth
                if self.use_random_agent and self.last_coverage_rand:
                    # For random agent, get depth from the selected question's score
                    selected_idx = self.last_coverage_rand.real_questions.index(self.last_selected_question)
                    depth_level = self.last_coverage_rand.question_scores[selected_idx].depth_score
                elif self.last_coverage:
                    # For standard agent, get depth from the coverage output
                    depth_level = self.last_coverage.depth_score
                
                self.section_coverage[danal.matched_section_id][danal.matched_question_id].append({
                    "coverage": danal.coverage_assessment,
                    "depth": depth_level
                })
                print(f"\nMatched to outline question: {danal.matched_question_id} in section {danal.matched_section_id}")
                print(f"Coverage: {danal.coverage_assessment:.2f}, Depth: {depth_level}")
            else:
                # No direct match - treat as deep dive
                is_deep_dive = True
                # Find the most recent matched question in the transcript
                if transcript and len(transcript) >= 2:
                    last_question = transcript[-2].text
                    for section in self.sections:
                        for question in section.questions:
                            if question.question.lower() in last_question.lower():
                                # Create deep dive question
                                deep_dive_id = f"D{len(self.deep_dive_section.questions) + 1}"
                                deep_dive_question = OutlineItem(
                                    id=deep_dive_id,
                                    question=last_question,
                                    importance="high"  # Deep dives are always high importance
                                )
                                self.deep_dive_section.questions.append(deep_dive_question)
                                depth_level = self.last_coverage.depth_score if self.last_coverage else 2
                                self.section_coverage["D"][deep_dive_id] = [{
                                    "coverage": danal.coverage_assessment,
                                    "depth": depth_level
                                }]
                                print(f"\nDeep dive detected! Added as {deep_dive_id}")
                                print(f"Original question: {last_question}")
                                print(f"Coverage: {danal.coverage_assessment:.2f}, Depth: {depth_level}")
                                break
        
        print(f"\nMatched: {danal.matched_question_id} in {danal.matched_section_id} (confidence: {danal.match_confidence:.2f})")
        print(f"Coverage assessment: {danal.coverage_assessment:.2f}")
        print(f"Emotional Insights: {danal.emotional_insights}")
        print(f"Chain of Thought: {danal.chain_of_thought}")
        print(f"Uncovered Angles: {danal.uncovered_angles}")
        print(f"Follow-up Questions: {danal.followup_questions}")
        print(f'Memory_snippets: {danal.memory_snippets}\n')

        if self.use_rag and self.session_id:
            # Update topic coverage
            if danal.matched_section_id and danal.matched_question_id:
                section = next(s for s in self.sections if s.id == danal.matched_section_id)
                question = next(q for q in section.questions if q.id == danal.matched_question_id)
                
                # Get depth from the agent's evaluation
                depth_level = 1  # Default depth
                if self.use_random_agent and self.last_coverage_rand:
                    # For random agent, get depth from the selected question's score
                    selected_idx = self.last_coverage_rand.real_questions.index(self.last_selected_question)
                    depth_level = self.last_coverage_rand.question_scores[selected_idx].depth_score
                elif self.last_coverage:
                    # For standard agent, get depth from the coverage output
                    depth_level = self.last_coverage.depth_score
                
                self.rag_memory.update_topic_coverage(
                    self.session_id,
                    topic_id=danal.matched_question_id,
                    topic_name=question.question,
                    depth_level=depth_level,
                    coverage_score=danal.coverage_assessment
                )
            # Update emotional states
            if danal.emotional_insights:
                self.rag_memory.add_emotional_state(
                    self.session_id, round_num, danal.emotional_insights[0], 1.0, response
                )
            # Add follow-up opportunities
            for fq in danal.followup_questions:
                self.rag_memory.add_followup_opportunity(
                    self.session_id, round_num, "exploratory", fq, 1.0
                )
            # Add memory snippets
            if danal.memory_snippets:
                self.rag_memory.add_memory_snippet(
                    self.session_id, round_num, danal.memory_snippets[0]
                )

        print("\n==== CONVERGENT THINKING WITH V2's CONVERGENT AGENT ====")
        
        # Convert V1's followup_questions to V2's DivergentQuestion format
        divergent_questions: List[DivergentQuestion] = []
        for i, fq in enumerate(danal.followup_questions):
            # Parse question from "Type: Question" format if needed
            if ":" in fq:
                parts = fq.split(":", 1)
                question_text = parts[1].strip() if len(parts) > 1 else fq
                question_type = parts[0].strip()
            else:
                question_text = fq
                question_type = "balanced"
            
            divergent_questions.append(DivergentQuestion(
                question=question_text,
                reasoning=f"Generated from divergent analysis based on: {', '.join(danal.uncovered_angles[:2]) if danal.uncovered_angles else 'conversation flow'}",
                source_agent_name="DAgent"
            ))
        
        # If no questions were generated, create a default one
        if not divergent_questions:
            # Create a default question based on outline
            default_question_text = "Can you tell me more about that?"
            if transcript and len(transcript) >= 2:
                # Try to create a question related to the last question
                last_q = transcript[-2].text
                default_question_text = f"Following up on that, what else can you share?"
            
            divergent_questions.append(DivergentQuestion(
                question=default_question_text,
                reasoning="Default question generated when no follow-up questions were suggested",
                source_agent_name="DAgent"
            ))
        
        # Prepare context for V2's ConvergentAgent
        conv_input = (
            f"USER PREFERENCE FOR THIS INTERVIEW: '{self.user_preference}'\n\n"
            f"CANDIDATE QUESTIONS:\n"
            f"{json.dumps([q.model_dump() for q in divergent_questions], indent=2, ensure_ascii=False)}\n\n"
            f"LATEST CONVERSATION TURN:\n{response}\n\n"
            f"COVERAGE STATUS:\n{self._create_coverage_status_for_convergent(danal)}\n\n"
            f"EXPLORATION PRIORITIES:\n{self._create_exploration_priorities()}"
        )
        
        if is_deep_dive:
            conv_input += "\n\nNOTE: This is a deep dive question. Focus on exploring the current topic in depth rather than moving to new sections."
        
        print(f"ConvergentAgent context: {len(conv_input)} chars")
        
        if self.use_random_agent:
            # Still use CAgentRand for random agent mode to maintain compatibility
            c_ctx = (
                f"CURRENT RESPONSE:\n{response}\n\n"
                f"DIVERGENT ANALYSIS RESULTS:\n{json.dumps(danal.model_dump(), ensure_ascii=False, indent=2)}\n\n"
                f"UPDATED COVERAGE STATUS:\n{self._create_coverage_status_for_convergent(danal)}\n\n"
                f"EXPLORATION PRIORITIES:\n{self._create_exploration_priorities()}"
            )
            if is_deep_dive:
                c_ctx += "\n\nNOTE: This is a deep dive question. Focus on exploring the current topic in depth rather than moving to new sections."
            
            c_ctx += f"\n\nUSER PREFERENCE: {self.user_preference}\nOutput all the questions and the scores in strict json format\n"
            try:
                c_res = await Runner.run(CAgentRand, c_ctx)
                c_out = c_res.final_output_as(CoverageOutputRand)
                
                # Calculate probabilities and select question
                scores = [score.score for score in c_out.question_scores]
                novelty_scores = [score.novelty_score for score in c_out.question_scores]
                depth_scores = [score.depth_score for score in c_out.question_scores]
                
                # Calculate user preference scores based on question types
                user_preference_scores = [
                    1.0 if q_type == self.user_preference else 0.0 
                    for q_type in c_out.question_types
                ]
                
                probabilities = calculate_softmax_probabilities(
                    convergent_scores=scores, 
                    user_preference_scores=user_preference_scores,
                    novelty_scores=novelty_scores,
                    depth_scores=depth_scores,
                    temperature=self.temperature,
                    lambda_preference=self.lambda_preference
                )
                
                # Select question based on probabilities
                selected_question = select_question_by_probability(
                    c_out.real_questions,
                    probabilities
                )
       
                # Store the output and selected question
                self.last_coverage_rand = c_out
                self.last_selected_question = selected_question
                
                print("\n==== Question Selection Details ====")
                print("Raw Scores:")
                for q, s, n, d, t in zip(c_out.real_questions, scores, novelty_scores, depth_scores, c_out.question_types):
                    print(f"  {q} ({t}):")
                    print(f"    Base Score: {s:.3f}")
                    print(f"    Novelty: {n:.3f}")
                    print(f"    Depth: {d}")
                print("\nProbabilities:")
                for q, p in zip(c_out.real_questions, probabilities):
                    print(f"  {q}: {p:.3f}")
                print(f"\nSelected Question: {selected_question}")
                print(f"Strategy: {c_out.exploration_strategy}")
                print(f"Reasoning: {c_out.reasoning}")
                print(f"Temperature: {self.temperature}")
                print(f"User Preference: {self.user_preference}")
                print(f"Lambda (Preference): {self.lambda_preference}")
                
                # Track the selected question
                self.asked_question_texts.add(selected_question.lower().strip())
                
                return danal, c_out
            except Exception as e:
                print(f"\n==== DEBUG: CAgentRand Error ====")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                raise
        else:
            # Use V2's ConvergentAgent
            try:
                conv_res = await Runner.run(ConvergentAgent, conv_input)
                conv_output = conv_res.final_output_as(ConvergentAgentOutput)
                
                # Convert ConvergentAgentOutput to CoverageOutput for compatibility
                # Extract depth_score and novelty_score if available from the chosen question
                # For now, we'll use defaults since V2's convergent agent doesn't provide these directly
                depth_score = 2  # Default medium depth
                novelty_score = 0.5  # Default medium novelty
                
                # Map strategy to exploration_strategy
                strategy_mapping = {
                    "deepen_emotion": "within-section",
                    "expand_outline": "cross-section",
                    "follow_persona": "within-section",
                    "creative_pivot": "cross-section",
                    "balance-exploration": "balance-exploration"
                }
                exploration_strategy = strategy_mapping.get(conv_output.strategy, "within-section")
                
                # Create CoverageOutput for backward compatibility
                c_out = CoverageOutput(
                    next_question=conv_output.next_question,
                    reasoning=conv_output.reasoning,
                    exploration_strategy=exploration_strategy,
                    novelty_score=novelty_score,
                    depth_score=depth_score
                )
                
                self.last_coverage = c_out
                
                # Track the question we're about to ask
                self.asked_question_texts.add(c_out.next_question.lower().strip())
                
                print(f"Next Question: {c_out.next_question}")
                print(f"Strategy: {c_out.exploration_strategy}")
                print(f"Reasoning: {c_out.reasoning}")
                print(f"Chosen from: {conv_output.chosen_divergent_question.source_agent_name}")
                
                return danal, c_out
            except Exception as e:
                print(f"\n==== DEBUG: V2 ConvergentAgent Error ====")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")
                raise
    
    def _create_coverage_status_for_convergent(self, danal: DivergentAnalysis) -> str:
        """Create coverage status context for convergent agent"""
        lines = ["=== CURRENT COVERAGE STATUS ==="]
        
        # Overall progress
        total_q = sum(len(s.questions) for s in self.sections)
        covered_q = sum(
            1 for s in self.sections for q in s.questions 
            if self.section_coverage[s.id][q.id]
        )
        lines.append(f"Overall Progress: {covered_q}/{total_q} questions ({covered_q/total_q*100:.1f}%)")
        
        # Current section status
        if danal.matched_section_id:
            current_sec_coverage = self.get_section_coverage_percentage(danal.matched_section_id)
            current_sec = next(s for s in self.sections if s.id == danal.matched_section_id)
            lines.append(f"\nCurrent Section: {current_sec.section} ({danal.matched_section_id})")
            lines.append(f"Section Coverage: {current_sec_coverage:.1f}%")
            
            # List uncovered questions in current section
            uncovered_current = [
                q for q in current_sec.questions 
                if not self.section_coverage[danal.matched_section_id][q.id]
            ]
            if uncovered_current:
                lines.append("Uncovered in current section:")
                for q in uncovered_current[:3]:  # Top 3
                    lines.append(f"  - {q.id}: {q.question} ({q.importance})")
        
        # Other sections needing attention
        low_coverage_sections = [
            (s, self.get_section_coverage_percentage(s.id))
            for s in self.sections 
            if self.get_section_coverage_percentage(s.id) < 50
        ]
        low_coverage_sections.sort(key=lambda x: x[1])  # Sort by coverage
        
        if low_coverage_sections:
            lines.append(f"\nSections Needing Attention (coverage < 50%):")
            for sec, coverage in low_coverage_sections[:3]:
                lines.append(f"  - {sec.section}: {coverage:.1f}%")
        
        return "\n".join(lines)
    
    def _create_exploration_priorities(self) -> str:
        """Create exploration priorities for convergent agent"""
        lines = ["=== EXPLORATION PRIORITIES ==="]
        
        # High priority uncovered questions
        high_priority = self.get_high_priority_uncovered_questions()[:5]
        if high_priority:
            lines.append("High Priority Uncovered:")
            for sec, q in high_priority:
                lines.append(f"  - {sec.section}: {q.question} ({q.importance})")
        
        # Sections that need more exploration
        section_priorities = []
        for sec in self.sections:
            coverage = self.get_section_coverage_percentage(sec.id)
            importance_weight = sum(
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(q.importance.lower(), 1)
                for q in sec.questions if not self.section_coverage[sec.id][q.id]
            )
            priority_score = importance_weight * (100 - coverage) / 100
            section_priorities.append((sec, coverage, priority_score))
        
        section_priorities.sort(key=lambda x: x[2], reverse=True)
        
        lines.append("\nSection Exploration Priority:")
        for sec, coverage, priority in section_priorities[:3]:
            lines.append(f"  - {sec.section}: {coverage:.1f}% coverage, priority score: {priority:.1f}")
        
        return "\n".join(lines)

    def get_coverage_report(self):
        """Generate a detailed coverage report."""
        report = []
            
        for section in self.sections:
            section_report = {
                "section_id": section.id,
                "section_name": section.section,
                "questions": []
            }
                
            for question in section.questions:
                scores = self.section_coverage[section.id][question.id]
                avg_coverage = sum(s["coverage"] for s in scores) / len(scores) if scores else 0.0
                avg_depth = sum(s["depth"] for s in scores) / len(scores) if scores else 0.0
                status = "Covered" if avg_coverage > 0.6 else "Partially Covered" if avg_coverage > 0.3 else "Missing"
                    
                section_report["questions"].append({
                    "id": question.id,
                    "question": question.question,
                    "importance": question.importance,
                    "status": status,
                    "coverage_score": round(avg_coverage, 2),
                    "depth_score": round(avg_depth, 2),
                    "mentions": len(scores)
                })
            
            report.append(section_report)        
        return report