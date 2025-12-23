import json
import os
import asyncio
import argparse
import sys
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
from collections import defaultdict


sys.path.insert(0, os.path.abspath('dc_agents'))

import openai

# --- Dynamic Imports for different versions ---
from dc_agents.V1.modelsv1 import ConversationTurn as ConversationTurnV1, OutlineSection as OutlineSectionV1, Persona as PersonaV1, load_outline_from_sections as load_outline_from_sections_v1
from dc_agents.V1.DCAgentV1 import DCAgentSystem as DCAgentSystemV1
from dc_agents.V2.modelsv2 import ConversationTurn as ConversationTurnV2, OutlineSection as OutlineSectionV2, Persona as PersonaV2, load_outline_from_sections as load_outline_from_sections_v2
from dc_agents.V2.DCAgentV2 import DCAgentSystemV2
from dc_agents.V3.modelsv3 import ConversationTurn as ConversationTurnV3, OutlineSection as OutlineSectionV3, Persona as PersonaV3, load_outline_from_sections as load_outline_from_sections_v3
from dc_agents.V3.DCAgentV3 import DCAgentSystemV3
from dc_agents.V1.rag_memory import RAGMemorySystem


# --- CONFIG ---
OPENAI_API_KEY = os.getenv("API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = os.getenv("BASE_URL")

# --- Base LLM Question Generator ---

class BaseLLMQuestionGenerator:
    """Base class for generating questions using an OpenAI-compatible API."""
    def __init__(self, model=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL):
        self.model = model
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url if base_url else None)
        self.source_name = "llm_base"

    async def generate_questions(self, transcript: List, outline: List, persona: 'Persona', num_questions: int = 3, **kwargs) -> List[str]:
        prompt = self._create_prompt(transcript, outline, persona, num_questions, **kwargs)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            return self._parse_response(content)
        except Exception as e:
            print(f"Error generating questions from {self.source_name}: {e}")
            return []

    def _create_prompt(self, transcript: List, outline: List, persona: 'Persona', num_questions: int, **kwargs) -> str:
        history = "\n".join([f"{turn.speaker}: {turn.text}" for turn in transcript[-6:]])
        outline_str = "\n".join([f"[{s.id}] {s.section}: " + ", ".join(q.question for q in s.questions) for s in outline])
        persona_str = f"Name: {persona.name}, Age: {persona.age}, Occupation: {persona.employment}, Personality: {persona.personality}"
        
        return f"""
You are a professional interviewer. Based on the following information, please generate {num_questions} suitable next questions for the interview.

[Interviewee Information]
{persona_str}

[Interview Outline]
{outline_str}

[Conversation History]
{history}

Please output the list of questions in JSON array format, for example:
["Question 1", "Question 2", "Question 3"]
"""

    def _parse_response(self, content: str) -> List[str]:
        import re
        def extract_questions_from_string(s: str) -> List[str]:
            # Remove code block markers
            s_clean = re.sub(r"```[a-zA-Z]*", "", s)
            s_clean = s_clean.replace("```", "")
            # Try to find a JSON array block
            array_match = re.search(r"\[.*?\]", s_clean, re.DOTALL)
            if array_match:
                array_str = array_match.group(0)
                try:
                    data = json.loads(array_str)
                    if isinstance(data, list):
                        return [str(q) for q in data]
                except Exception:
                    pass  # If parsing fails, fallback
            # Fallback: collect lines that look like questions
            lines = [line.strip() for line in s_clean.split("\n") if line.strip()]
            question_lines = [l for l in lines if l.startswith("\"") or l.startswith("-") or re.match(r"^\d+\. ", l)]
            # Remove leading dash, number, or quotes
            cleaned = [re.sub(r'^("|-|\d+\.)\s*', '', l).strip('"') for l in question_lines]
            return [q for q in cleaned if q]

        # If content is a list, process each string
        if isinstance(content, list):
            for s in content:
                qs = extract_questions_from_string(s)
                if qs:
                    return qs
            # Fallback: flatten all question-like lines
            all_lines = []
            for s in content:
                all_lines.extend(extract_questions_from_string(s))
            return all_lines
        # If content is a string, process as before
        if isinstance(content, str):
            # Try to parse as JSON directly
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return [str(q) for q in data]
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, list):
                            return [str(q) for q in value]
            except Exception:
                pass
            # Otherwise, try to extract from string
            qs = extract_questions_from_string(content)
            if qs:
                return qs
            # Fallback: collect all non-empty lines
            return [line.strip() for line in content.split("\n") if line.strip()]
        # If content is neither string nor list, fallback to string conversion
        return [str(content)]


# --- Enhanced LLM Generators ---

class CoTLLMQuestionGenerator(BaseLLMQuestionGenerator):
    """Generates questions using a Chain-of-Thought prompt."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_name = "llm_cot"

    def _create_prompt(self, transcript: List, outline: List, persona: 'Persona', num_questions: int, **kwargs) -> str:
        base_prompt = super()._create_prompt(transcript, outline, persona, num_questions, **kwargs)
        return f"""{base_prompt}
First, think step-by-step about the interview's goal, the interviewee's personality, and the recent conversation flow. Consider what topics are yet to be covered and what previous points could be explored deeper. Based on this reasoning, then generate the questions.
Please output the list of questions in JSON array format, for example:
["Question 1", "Question 2", "Question 3"]
"""

class ToTLLMQuestionGenerator(BaseLLMQuestionGenerator):
    """Generates questions using a standard Tree-of-Thought prompt."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_name = "llm_tot"

    def _create_prompt(self, transcript: List, outline: List, persona: 'Persona', num_questions: int, **kwargs) -> str:
        base_prompt = super()._create_prompt(transcript, outline, persona, num_questions, **kwargs)
        return f"""{base_prompt}
Explore multiple reasoning paths to decide on the best questions.
1.  Path 1: Focus on deepening the last topic discussed.
2.  Path 2: Focus on transitioning to a new, relevant topic from the outline.
3.  Path 3: Focus on the interviewee's emotional state or a surprising remark they made.
Evaluate these paths and generate a final list of questions that synthesizes the best options.
Please output the list of questions in JSON array format, for example:
["Question 1", "Question 2", "Question 3"]
"""

class RAGLLMQuestionGenerator(BaseLLMQuestionGenerator):
    """Generates questions using a RAG-based approach."""
    def __init__(self, rag_system: RAGMemorySystem, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_system = rag_system
        self.source_name = "llm_rag"

    def _create_prompt(self, transcript: List, outline: List, persona: 'Persona', num_questions: int, **kwargs) -> str:
        session_id = kwargs.get("session_id")
        current_query = "\n".join([f"{turn.speaker}: {turn.text}" for turn in transcript[-2:]])
        
        # Retrieve relevant memories
        retrieved_snippets = []
        if session_id:
            similar_turns = self.rag_system.search_similar_conversations(query=current_query, session_id=session_id, top_k=3)
            retrieved_snippets = [f"Round {t['round_num']} ({t['speaker']}): {t['content']}" for t in similar_turns]

        retrieved_context = "\n".join(retrieved_snippets)
        base_prompt = super()._create_prompt(transcript, outline, persona, num_questions, **kwargs)
        
        return f"""{base_prompt}

[Retrieved from long-term memory]
Here are some potentially relevant snippets from earlier in the conversation:
{retrieved_context if retrieved_context else "No relevant memories found."}

Based on the conversation history AND the retrieved memories, generate the next questions.
Please output the list of questions in JSON array format, for example:
["Question 1", "Question 2", "Question 3"]
"""


@dataclass
class SuggestedQuestion:
    """Represents a single suggested question with its metadata."""
    question: str
    source: str  # 'dcagent_vX_AGENTNAME', 'llm_base', 'llm_cot', etc.
    is_selected: bool
    agent_specific_data: Optional[Dict] = None

@dataclass
class SurveyItem:
    """Represents a single item in the survey data, containing context and suggested questions."""
    round_num: int
    previous_question: str  # The question that was just asked
    answer: str            # The answer that was just given
    suggested_questions: List[SuggestedQuestion]  # Questions suggested for the next round
    agent_specific_data: Optional[Dict] = None

class ComparisonRunner:
    """
    Generates comparison data by processing a transcript and collecting question suggestions
    from DCAgent v1, v2, v3, and multiple LLM strategies.
    """
    def __init__(self, transcript_path: str, outline_path: str, persona_path: str, only_v1: bool = False):
        self.transcript_path = transcript_path
        self.outline_path = outline_path
        self.persona_path = persona_path
        self.only_v1 = only_v1
        self.session_id = f"comparison_run_{uuid.uuid4()}"
        
        # Load base data
        with open(transcript_path, "r", encoding="utf-8") as f:
            self.transcript_data = json.load(f)["transcript"]
        
        self.survey_data: List[SurveyItem] = []
        self.rag_system = RAGMemorySystem(db_path=f"data/rag/{self.session_id}.db")

        # --- Initialize all systems ---
        self.systems = {}
        self._init_dc_agents()
        if not only_v1:
            self._init_llm_generators()
        else:
            self.llm_generators = {}

    def _init_dc_agents(self):
        # V1
        persona_v1 = PersonaV1(**json.load(open(self.persona_path))["persona"])
        sections_v1 = load_outline_from_sections_v1(self.outline_path)
        self.systems['v1'] = DCAgentSystemV1(sections=sections_v1, persona=persona_v1, session_id=f"{self.session_id}_v1")
        # V2
        persona_v2 = PersonaV2(**json.load(open(self.persona_path))["persona"])
        sections_v2 = load_outline_from_sections_v2(self.outline_path)
        self.systems['v2'] = DCAgentSystemV2(sections=sections_v2, persona=persona_v2, session_id=f"{self.session_id}_v2")
        # V3
        persona_v3 = PersonaV3(**json.load(open(self.persona_path))["persona"])
        sections_v3 = load_outline_from_sections_v3(self.outline_path)
        self.systems['v3'] = DCAgentSystemV3(sections=sections_v3, persona=persona_v3, session_id=f"{self.session_id}_v3")

    def _init_llm_generators(self):
        self.llm_generators = {
            'llm_base': BaseLLMQuestionGenerator(),
            'llm_cot': CoTLLMQuestionGenerator(),
            'llm_tot': ToTLLMQuestionGenerator(),
            'llm_rag': RAGLLMQuestionGenerator(self.rag_system),
        }

    async def generate_data(self):
        """
        Processes the transcript turn-by-turn to generate and collect questions from all systems.
        """
        print(f"\n=== Starting Comparison Data Generation for Session: {self.session_id} ===")
        
        # Use a generic ConversationTurn for RAG history
        transcript_for_rag = [ConversationTurnV2(**turn) for turn in self.transcript_data]
        for i, turn_data in enumerate(self.transcript_data):
            turn = ConversationTurnV2(**turn_data)
            self.rag_system.add_conversation_turn(self.session_id, i // 2, turn)
        
        for i in range(0, len(self.transcript_data) - 1, 2):
            prev_q_turn_data = self.transcript_data[i]
            prev_a_turn_data = self.transcript_data[i+1]
            round_num = i // 2
            print(f"\n[Round {round_num}] Processing...")

            all_suggestions: List[SuggestedQuestion] = []
            question_texts: Set[str] = set()
            
            # This dictionary will hold agent-specific data for the round
            agent_round_data = defaultdict(dict)

            # --- Get suggestions from DC Agents ---
            versions_to_run = ['v1'] if self.only_v1 else ['v1', 'v2', 'v3']
            for version in versions_to_run:
                try:
                    dc_system = self.systems[version]
                    # Need to construct version-specific objects for each system
                    if version == 'v1':
                        transcript_so_far = [ConversationTurnV1(**t) for t in self.transcript_data[:i+2]]
                        answer_text = ConversationTurnV1(**prev_a_turn_data).text
                        analysis_data, convergent_output = await dc_system.process_response(
                            response=answer_text, round_num=round_num, transcript=transcript_so_far
                        )
                    elif version == 'v2':
                        transcript_so_far = [ConversationTurnV2(**t) for t in self.transcript_data[:i+2]]
                        answer_text = ConversationTurnV2(**prev_a_turn_data).text
                        analysis_data, convergent_output = await dc_system.process_response(
                            answer_text, transcript_so_far
                        )
                    else: # v3
                        transcript_so_far = [ConversationTurnV3(**t) for t in self.transcript_data[:i+2]]
                        answer_text = ConversationTurnV3(**prev_a_turn_data).text
                        analysis_data, convergent_output = await dc_system.process_response(
                            answer_text, transcript_so_far
                        )

                    # --- Store architecturally specific data ---
                    if version == 'v1':
                        agent_round_data['dcagent_v1']['option_set'] = analysis_data.followup_questions
                    elif version == 'v2':
                        agent_round_data['dcagent_v2']['option_set'] = [q.model_dump() for q in analysis_data.get("edited_questions", [])]
                        agent_round_data['dcagent_v2']['winning_specialist'] = convergent_output.chosen_divergent_question.source_agent_name
                    elif version == 'v3':
                        agent_round_data['dcagent_v3']['option_set'] = [q.model_dump() for q in analysis_data.get("edited_questions", [])]
                        agent_round_data['dcagent_v3']['divergent_plan'] = analysis_data.get("divergent_plan").model_dump()
                        agent_round_data['dcagent_v3']['winning_specialist'] = convergent_output.chosen_divergent_question.source_agent_name

                    selected_question_text = convergent_output.next_question

                    if version == 'v1':
                        dc_agent_questions = analysis_data.followup_questions
                        champion_found_in_list = False
                        for q_text in dc_agent_questions:
                            if q_text and q_text not in question_texts:
                                # V1's convergent question is clean, but divergent questions have prefixes.
                                # Thus, we must check for substring inclusion.
                                is_champion = False
                                if selected_question_text and selected_question_text in q_text:
                                    is_champion = True
                                    champion_found_in_list = True
                                
                                all_suggestions.append(SuggestedQuestion(
                                    question=q_text,
                                    source='dcagent_v1_DAgent',
                                    is_selected=is_champion
                                ))
                                question_texts.add(q_text)
                        
                        # Fallback in case the convergent champion isn't a substring of any divergent one.
                        if selected_question_text and not champion_found_in_list and selected_question_text not in question_texts:
                             all_suggestions.append(SuggestedQuestion(
                                question=selected_question_text,
                                source='dcagent_v1_convergent_champion',
                                is_selected=True
                            ))
                             question_texts.add(selected_question_text)
                    else: # V2 and V3
                        # V2/V3 return a dict with a list of DivergentQuestion objects
                        dc_agent_questions = analysis_data.get("all_divergent_questions", [])
                        champion_found_in_list = False
                        for div_q in dc_agent_questions:
                            q_text = div_q.question
                            if q_text and q_text not in question_texts:
                                is_champion = (q_text == selected_question_text)
                                if is_champion:
                                    champion_found_in_list = True
                                all_suggestions.append(SuggestedQuestion(
                                    question=q_text,
                                    source=f'dcagent_{version}_{div_q.source_agent_name}',
                                    is_selected=is_champion
                                ))
                                question_texts.add(q_text)

                        # If the selected champion from convergent_output was not in the divergent list, add it.
                        if selected_question_text and not champion_found_in_list and selected_question_text not in question_texts:
                            all_suggestions.append(SuggestedQuestion(
                                question=selected_question_text,
                                source=f'dcagent_{version}_convergent_champion', # Add a specific source
                                is_selected=True
                            ))
                            question_texts.add(selected_question_text)
                except Exception as e:
                    import traceback
                    print(f"  ! Error getting questions from DCAgent {version}: {e}")
                    traceback.print_exc()

            # --- Get suggestions from LLM Generators ---
            if not self.only_v1:
                # Use V2 models for LLM generators as a common ground
                transcript_for_llms = [ConversationTurnV2(**t) for t in self.transcript_data[:i+2]]
                persona_for_llms = PersonaV2(**json.load(open(self.persona_path))["persona"])
                outline_for_llms = load_outline_from_sections_v2(self.outline_path)

                for name, generator in self.llm_generators.items():
                    try:
                        llm_questions = await generator.generate_questions(
                            transcript_for_llms, outline_for_llms, persona_for_llms, 
                            num_questions=4, session_id=self.session_id
                        )
                        # Store the full list as the "option set" for baselines
                        agent_round_data[name]['option_set'] = llm_questions

                        for idx, q_text in enumerate(llm_questions):
                            if q_text and q_text not in question_texts:
                                is_selected = (idx == 0) # First question is the selected one for LLMs
                                all_suggestions.append(SuggestedQuestion(
                                    question=q_text,
                                    source=name,
                                    is_selected=is_selected
                                ))
                                question_texts.add(q_text)
                    except Exception as e:
                        print(f"  ! Error getting questions from {name}: {e}")

            self.survey_data.append(
                SurveyItem(
                    round_num=round_num,
                    previous_question=prev_q_turn_data["text"],
                    answer=prev_a_turn_data["text"],
                    suggested_questions=[
                        SuggestedQuestion(
                            question=sq.question,
                            source=sq.source,
                            is_selected=sq.is_selected,
                            # Attach the round-specific data to the first suggestion from that source
                            agent_specific_data=agent_round_data.get( (sq.source.split('_')[0] if 'dcagent' in sq.source else sq.source) )
                        )
                        for sq in all_suggestions
                    ]
                )
            )
            print(f"  > Generated {len(all_suggestions)} unique questions in total.")

    def save_data(self, output_path: str):
        """Saves the collected survey data to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_list = [asdict(item) for item in self.survey_data]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"\nComparison data saved to {output_path}")


async def main():
    """Main function to run the comparison data generation process."""
    parser = argparse.ArgumentParser(description='Generate comparison data from an interview transcript across multiple agent versions.')
    parser.add_argument('--transcript-path', type=str, default='data/transcripts/mermaid_real.json',
                        help='Path to the input transcript JSON file.')
    parser.add_argument('--outline-path', type=str, default='data/outlines/example_outline.json',
                        help='Path to the outline JSON file.')
    parser.add_argument('--persona-path', type=str, default='data/personas/example_persona.json',
                        help='Path to the persona JSON file.')
    parser.add_argument('--only-v1', action='store_true',
                        help='Only generate questions for V1 (skip v2, v3, and LLM baselines).')
    
    args = parser.parse_args()

    output_filename = f"comparison_data_enhanced_{os.path.basename(args.transcript_path)}"
    
    runner = ComparisonRunner(
        transcript_path=args.transcript_path,
        outline_path=args.outline_path,
        persona_path=args.persona_path,
        only_v1=args.only_v1,
    )
    await runner.generate_data()
    runner.save_data(f"data/evaluations/{output_filename}")

if __name__ == "__main__":
    asyncio.run(main()) 