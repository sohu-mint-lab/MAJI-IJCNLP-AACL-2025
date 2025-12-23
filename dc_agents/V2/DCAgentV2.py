import asyncio
import json
import os
import warnings
from typing import List, Optional, Tuple, Dict

from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled, set_default_openai_api
from openai import AsyncOpenAI

from .modelsv2 import (
    ConversationTurn, OutlineSection, Persona, BackgroundSummary, KeywordsOutput,
    OutlineMatch, DivergentAgentOutput, ConvergentAgentOutput, DivergentQuestion,
    flatten_outline_sections
)

# --- Environment and Client Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

# === AGENT DEFINITIONS ===

BackgroundAgent = Agent(
    name="BackgroundSummarizerAgent",
    instructions=(
        "You are an AI assistant that maintains a dynamic background summary for an ongoing interview. "
        "Your task is to integrate the latest conversation turn into the existing background summary. "
        "The persona of the interviewee is provided for context.\n\n"
        "RULES:\n"
        "1.  **Update, Don't Replace**: Append and blend the new information, refining the summary. "
        "2.  **Long-Term Section (approx. 150 words)**: This section should hold the most critical, foundational information about the interviewee. Condense and merge key details from the 'short-term' section over time if they prove to be significant.\n"
        "3.  **Short-Term Section (approx. 50 words)**: This section should capture the essence of the most recent 1-3 conversation turns. It's more volatile and focuses on the immediate context.\n"
        "4.  **Persona-Aware**: Consider the interviewee's persona to judge the importance of information.\n"
        "5.  **Output**: You must output a JSON object with 'long_term_summary' and 'short_term_summary' fields."
    ),
    model="gpt-4.1-mini",
    output_type=BackgroundSummary,
)

KeywordsAgent = Agent(
    name="KeywordsExtractorAgent",
    instructions=(
        "You are an AI assistant that extracts critical keywords from the latest conversation turn. "
        "Use the provided background summary to identify keywords that are not only salient to the current turn but also connect to the broader conversation context.\n\n"
        "RULES:\n"
        "1.  **Focus**: Analyze the 'CURRENT CONVERSATION TURN' primarily.\n"
        "2.  **Contextualize**: Use the 'BACKGROUND SUMMARY' to find resonant or recurring themes.\n"
        "3.  **Extract, Don't Invent**: All keywords must originate from the text.\n"
        "4.  **Output**: Provide a JSON list of strings, e.g., {\"keywords\": [\"keyword1\", \"keyword2\"]}"
    ),
    model="gpt-4.1-mini",
    output_type=KeywordsOutput,
)

OutlineMatcherAgent = Agent(
    name="OutlineMatcherAgent",
    instructions=(
        "You are a precise AI analyst. Your sole job is to match the user's latest response to a specific question in the provided interview outline. "
        "You must determine the best match and assess how well the response covers the question.\n\n"
        "**Matching Rules**:\n"
        "-   Directly answers an outline question: confidence 0.8+, coverage 0.7+\n"
        "-   Partially relevant: confidence 0.5-0.8, coverage 0.3-0.7\n"
        "-   Indirectly mentions concepts: confidence 0.3-0.5, coverage 0.1-0.3\n"
        "-   Irrelevant: confidence < 0.3, coverage 0.0\n\n"
        "**Output Format**:\n"
        "You must output a single JSON object with the following fields:\n"
        "-   `matched_question_id`: The ID of the most relevant question (e.g., 'S1Q2') or null.\n"
        "-   `matched_section_id`: The section ID of the matched question (e.g., 'S1') or null.\n"
        "-   `match_confidence`: A float from 0.0 to 1.0.\n"
        "-   `coverage_assessment`: A float from 0.0 to 1.0."
    ),
    model="gpt-4.1-mini",
    output_type=OutlineMatch,
)

# --- Divergent Agents ---

DIVERGENT_COMMON_INSTRUCTIONS = (
    "You are a creative and insightful interview question generator. Your goal is to propose at least one, and up to three, thoughtful follow-up questions based on the provided context. "
    "For each question, provide your reasoning. The question must be a string, and the reasoning must be a string."
    "The `source_agent_name` must be your agent name."
    "Your output **MUST** be a valid JSON object adhering to the `DivergentAgentOutput` schema."
    "Do not simply repeat questions from the outline."
)

ChainOfThoughtDivergentAgent = Agent(
    name="ChainOfThoughtDivergentAgent",
    instructions=(
        f"{DIVERGENT_COMMON_INSTRUCTIONS}\n\n"
        "**Your Specialization: Logic and Causality**\n"
        "Focus on the 'why' and 'how'. Analyze the logical flow of the conversation. Ask questions that uncover motivations, processes, and consequences.\n "
        "Connect ideas that were mentioned but not explicitly linked.\n"
        "Do chain of thought reasoning for each question."
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)

EmotionDivergentAgent = Agent(
    name="EmotionDivergentAgent",
    instructions=(
        f"{DIVERGENT_COMMON_INSTRUCTIONS}\n\n"
        "**Your Specialization: Emotional Depth**\n"
        "Focus on the feelings and emotions behind the words. Ask questions that explore the interviewee's emotional state, values, and personal significance of their experiences. "
        "Listen for subtext and unspoken feelings."
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)

OutlineDivergentAgent = Agent(
    name="OutlineDivergentAgent",
    instructions=(
        f"{DIVERGENT_COMMON_INSTRUCTIONS}\n\n"
        "**Your Specialization: Structured Progression**\n"
        "Your goal is to ensure the interview covers all essential topics from the outline. "
        "Ask questions that bridge the current conversation to uncovered, high-priority, or logically adjacent topics in the outline. "
        "Your questions should be inspired by the outline but phrased naturally in the context of the conversation.\n"
        "Think about how to expand on the outline while keeping the conversation flowing. If a question from the outline has already been asked, do not repeat it. "
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)

PersonaDivergentAgent = Agent(
    name="PersonaDivergentAgent",
    instructions=(
        f"{DIVERGENT_COMMON_INSTRUCTIONS}\n\n"
        "**Your Specialization: Role-playing**\n"
        "Think from the interviewee's perspective. Based on their persona (background, personality, goals), what question would they find most engaging or relevant? "
        "Ask questions that resonate with their stated experiences and character."
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)

NoveltyDivergentAgent = Agent(
    name="NoveltyDivergentAgent",
    instructions=(
        f"{DIVERGENT_COMMON_INSTRUCTIONS}\n\n"
        "**Your Specialization: Creative Surprise**\n"
        "Your goal is to introduce novel angles and break patterns. Ask questions that are unexpected but still relevant. "
        "Think about metaphors, hypothetical scenarios, or connections to broader themes that haven't been touched upon. Challenge assumptions."
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)

EditorAgent = Agent(
    name="EditorAgent",
    instructions=(
        "You are an expert editor. Your task is to review a list of proposed interview questions from different AI agents. "
        "Your goal is to clean up this list by removing duplicates and combining very similar questions.\n\n"
        "RULES:\n"
        "1.  **Identify Similarity**: Read through all questions and group them by topic or intent.\n"
        "2.  **Deduplicate**: If questions are nearly identical, keep only one.\n"
        "3.  **Merge & Refine**: If questions are very similar but have unique angles, synthesize them into a single, more comprehensive question. If they are distinct enough, keep both.\n"
        "4.  **Preserve Source**: When you keep or merge a question, you MUST preserve the `source_agent_name` and `reasoning` of the strongest original question in that cluster.\n"
        "5.  **Output**: You must return a single JSON object conforming to the `DivergentAgentOutput` schema, containing the cleaned and deduplicated list of questions. The `source_agent_name` for each question in your output MUST be the name of the original agent that created it."
    ),
    model="gpt-4.1-mini",
    output_type=DivergentAgentOutput,
)


# --- Convergent Agent ---

ConvergentAgent = Agent(
    name="ConvergentAgent",
    instructions=(
        "You are the Editor-in-Chief of this interview, responsible for selecting the single best question to ask next. "
        "You will be given a list of candidate questions from various specialist agents. Your decision should be guided by the user's stated preference for the interview's direction.\n\n"
        "**Your Task**:\n"
        "1.  **Review all suggestions**: Evaluate the questions proposed by the divergent agents.\n"
        "2.  **Consult User Preference**: The user's goal (e.g., 'focus on emotional depth', 'prioritize outline coverage') is your primary guide.\n"
        "3.  **Select the Winner**: Choose the single best question that aligns with the preference and the flow of conversation.\n"
        "4.  **Justify Your Choice**: Provide a clear reasoning for your selection, explaining why it's superior to the others in light of the user's preference.\n"
        "5.  **Define Your Strategy**: Name the strategy you employed (e.g., 'deepen_emotion', 'expand_outline', 'follow_persona', 'creative_pivot').\n"
        "6.  **Include Chosen Question**: You MUST include the complete `chosen_divergent_question` object that contains the question, reasoning, and source_agent_name of the selected question.\n\n"
        "**Output Format**:\n"
        "Your output **MUST** be a single, valid JSON object with these fields:\n"
        "- `next_question`: The selected question as a string\n"
        "- `reasoning`: Your reasoning for selecting this question\n"
        "- `strategy`: The strategy you employed\n"
        "- `chosen_divergent_question`: The complete DivergentQuestion object that was selected (including question, reasoning, and source_agent_name fields)"
    ),
    model="gpt-4.1-mini",
    output_type=ConvergentAgentOutput,
)


class DCAgentSystemV2:
    def __init__(
        self,
        sections: List[OutlineSection],
        persona: Persona,
        session_id: str | None = None,
        user_preference: str = "balanced", # e.g., "emotion", "outline", "novelty"
    ):
        self.sections = sections
        self.persona = persona
        self.session_id = session_id
        self.user_preference = user_preference

        self.background_summary = BackgroundSummary(
            long_term_summary="No long-term summary has been generated yet.",
            short_term_summary="The interview is just beginning."
        )

        self.section_coverage: Dict[str, Dict[str, List[Dict[str, float]]]] = {
            s.id: {q.id: [] for q in s.questions} for s in self.sections
        }
        
        self.divergent_agents = [
            ChainOfThoughtDivergentAgent,
            EmotionDivergentAgent,
            OutlineDivergentAgent,
            PersonaDivergentAgent,
            NoveltyDivergentAgent,
        ]
        
        self.last_analysis: Optional[Dict[str, any]] = None
        self.last_convergent_output: Optional[ConvergentAgentOutput] = None

    def _format_persona(self) -> str:
        p_dict = self.persona.model_dump()
        lines = ["--- INTERVIEWEE PERSONA ---"]
        for key, value in p_dict.items():
            if value:
                lines.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(lines)

    def _create_outline_context(self) -> str:
        """Creates a formatted string of the outline with coverage stats."""
        lines = ["=== INTERVIEW OUTLINE & COVERAGE ==="]
        for sec in self.sections:
            covered_q = sum(1 for q in sec.questions if self.section_coverage[sec.id][q.id])
            total_q = len(sec.questions)
            coverage_pct = (covered_q / total_q * 100) if total_q > 0 else 0
            lines.append(f"\n[{sec.id}] {sec.section} - Coverage: {coverage_pct:.1f}%")

            for q in sec.questions:
                scores = self.section_coverage[sec.id][q.id]
                mark = "V" if scores and (sum(s["coverage"] for s in scores) / len(scores)) > 0.6 else "X"
                lines.append(f"  {mark} {q.id}: {q.question} ({q.importance}, depth {q.depth})")
        return "\n".join(lines)

    async def process_response(
        self, response: str, transcript: List[ConversationTurn]
    ) -> Tuple[Dict[str, any], ConvergentAgentOutput]:
        """
        Runs the full V2 agent pipeline for one turn.
        """
        # --- STEP 1: Background Summary ---
        print("\n--- [Step 1/6] Updating Background Summary ---")
        bg_input = (
            f"PERSONA:\n{self._format_persona()}\n\n"
            f"EXISTING BACKGROUND SUMMARY:\n{self.background_summary.full_summary}\n\n"
            f"LATEST CONVERSATION TURN:\nSpeaker: {transcript[-1].speaker}, Text: {transcript[-1].text}"
        )
        bg_res = await Runner.run(BackgroundAgent, bg_input)
        self.background_summary = bg_res.final_output_as(BackgroundSummary)
        print(f"  [+] Background updated. Short-term: \"{self.background_summary.short_term_summary[:50]}...\"")
        
        # --- STEP 2: Keyword Extraction ---
        print("\n--- [Step 2/6] Extracting Keywords ---")
        kw_input = (
            f"BACKGROUND SUMMARY:\n{self.background_summary.full_summary}\n\n"
            f"CURRENT CONVERSATION TURN:\n{response}"
        )
        kw_res = await Runner.run(KeywordsAgent, kw_input)
        keywords = kw_res.final_output_as(KeywordsOutput)
        print(f"  [+] Keywords extracted: {keywords.keywords}")

        # --- STEP 3: Outline Matching ---
        print("\n--- [Step 3/6] Matching to Outline ---")
        om_input = (
            f"OUTLINE:\n{self._create_outline_context()}\n\n"
            f"USER RESPONSE TO MATCH:\n{response}"
        )
        om_res = await Runner.run(OutlineMatcherAgent, om_input)
        outline_match = om_res.final_output_as(OutlineMatch)
        
        if outline_match.matched_question_id:
            print(f"  [+] Response matched to: {outline_match.matched_question_id} (Confidence: {outline_match.match_confidence:.2f})")
            # Update coverage
            self.section_coverage.setdefault(outline_match.matched_section_id, {})
            self.section_coverage[outline_match.matched_section_id].setdefault(outline_match.matched_question_id, [])
            self.section_coverage[outline_match.matched_section_id][outline_match.matched_question_id].append(
                {"coverage": outline_match.coverage_assessment, "depth": 1} # Depth is harder to assess here, default to 1
            )
        else:
            print("  [+] Response did not match any outline question.")


        # --- STEP 4: Parallel Divergent Thinking ---
        print("\n--- [Step 4/6] Generating Divergent Questions (Parallel) ---")
        div_common_input = (
            f"PERSONA:\n{self._format_persona()}\n\n"
            f"BACKGROUND SUMMARY:\n{self.background_summary.full_summary}\n\n"
            f"EXTRACTED KEYWORDS: {json.dumps(keywords.keywords)}\n\n"
            f"LATEST CONVERSATION TURN:\n{response}\n\n"
            f"CURRENT OUTLINE COVERAGE:\n{self._create_outline_context()}"
        )
        print("Current Divergent Agent input size:", len(div_common_input))
        tasks = [Runner.run(agent, div_common_input) for agent in self.divergent_agents]
        divergent_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_divergent_questions: List[DivergentQuestion] = []
        for i, result in enumerate(divergent_results):
            agent_name = self.divergent_agents[i].name
            if isinstance(result, Exception):
                print(f"  [!] Agent {agent_name} failed: {result}")
                continue
            
            output = result.final_output_as(DivergentAgentOutput)
            all_divergent_questions.extend(output.questions)
            print(f"  [+] {agent_name} proposed {len(output.questions)} question(s).")


        # --- STEP 5: Editing and Deduplication ---
        print("\n--- [Step 5/6] Editing and Deduplicating Questions ---")
        if not all_divergent_questions:
            print("  [!] No divergent questions to edit. Skipping.")
            questions_for_convergent = []
        else:
            editor_input = (
                "Please review, deduplicate, and refine the following list of interview questions:\n\n"
                f"{json.dumps([q.model_dump() for q in all_divergent_questions], indent=2, ensure_ascii=False)}"
            )
            editor_res = await Runner.run(EditorAgent, editor_input)
            edited_output = editor_res.final_output_as(DivergentAgentOutput)
            questions_for_convergent = edited_output.questions
            print(f"  [+] Editor reduced {len(all_divergent_questions)} questions to {len(questions_for_convergent)}.")

        # --- STEP 6: Convergent Decision ---
        print("\n--- [Step 6/6] Making Convergent Decision ---")
        if not questions_for_convergent:
            raise ValueError("No questions were generated by any divergent agent or they were all filtered by the editor.")

        conv_input = (
            f"USER PREFERENCE FOR THIS INTERVIEW: '{self.user_preference}'\n\n"
            f"CANDIDATE QUESTIONS (pre-screened by an editor):\n"
            f"{json.dumps([q.model_dump() for q in questions_for_convergent], indent=2, ensure_ascii=False)}\n\n"
            f"LATEST CONVERSATION TURN:\n{response}\n\n"
        )
        print("Current Convergent Agent input size:", len(conv_input))
        conv_res = await Runner.run(ConvergentAgent, conv_input)
        convergent_output = conv_res.final_output_as(ConvergentAgentOutput)
        
        print(f"  [+] Convergent Agent selected question from {convergent_output.chosen_divergent_question.source_agent_name}")
        print(f"  [+] Strategy: {convergent_output.strategy}")
        print(f"  [+] Next Question: {convergent_output.next_question}")

        # --- FINALIZE ---
        self.last_analysis = {
            "background_summary": self.background_summary,
            "keywords": keywords,
            "outline_match": outline_match,
            "all_divergent_questions": all_divergent_questions,
            "edited_questions": questions_for_convergent,
        }
        self.last_convergent_output = convergent_output

        return self.last_analysis, self.last_convergent_output

    # --- Reporting Methods (for compatibility with playground) ---

    def get_section_summary(self):
        summ = {}
        for sec in self.sections:
            all_scores = [
                s["coverage"] for q in sec.questions if q.id in self.section_coverage.get(sec.id, {})
                for s in self.section_coverage[sec.id][q.id]
            ]
            covered = sum(bool(self.section_coverage.get(sec.id, {}).get(q.id)) for q in sec.questions)
            total_q = len(sec.questions)
            
            summ[sec.id] = {
                "section_name": sec.section,
                "avg_coverage": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
                "covered_questions": covered,
                "total_questions": total_q,
                "coverage_percentage": round(covered / total_q * 100, 1) if total_q > 0 else 0,
            }
        return summ

    def get_coverage_report(self):
        """Generates a detailed coverage report."""
        report = []
        for section in self.sections:
            section_report = {
                "section_id": section.id,
                "section_name": section.section,
                "questions": []
            }
            
            for question in section.questions:
                scores = self.section_coverage.get(section.id, {}).get(question.id, [])
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