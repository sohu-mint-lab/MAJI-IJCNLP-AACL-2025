import json
import os
import asyncio
import argparse
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, asdict

import openai

# --- Dynamic Imports based on version ---
def get_imports_for_version(version: str):
    """Dynamically imports the required classes based on the version."""
    if version == 'v1':
        from dc_agents.V1.modelsv1 import ConversationTurn, OutlineSection, Persona, load_outline_from_sections
        from dc_agents.V1.DCAgentV1 import DCAgentSystem as DCAgentSystemV1
        return DCAgentSystemV1, ConversationTurn, OutlineSection, Persona, load_outline_from_sections
    elif version == 'v2':
        from dc_agents.V2.modelsv2 import ConversationTurn, OutlineSection, Persona, load_outline_from_sections
        from dc_agents.V2.DCAgentV2 import DCAgentSystemV2
        return DCAgentSystemV2, ConversationTurn, OutlineSection, Persona, load_outline_from_sections
    elif version == 'v3':
        from dc_agents.V3.modelsv3 import ConversationTurn, OutlineSection, Persona, load_outline_from_sections
        from dc_agents.V3.DCAgentV3 import DCAgentSystemV3
        return DCAgentSystemV3, ConversationTurn, OutlineSection, Persona, load_outline_from_sections
    else:
        raise ValueError("Invalid version specified. Choose from 'v1', 'v2', 'v3'.")

# --- CONFIG ---
OPENAI_API_KEY = os.getenv("API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = os.getenv("BASE_URL")

class LLMQuestionGenerator:
    """Generates questions using a direct call to an OpenAI-compatible API."""
    def __init__(self, model=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL):
        self.model = model
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None
        )

    async def generate_questions(self, transcript: List, outline: List, persona: 'Persona', num_questions: int = 3) -> List[str]:
        """
        Generates a list of potential next questions based on the conversation history.
        """
        history = "\n".join([f"{turn.speaker}: {turn.text}" for turn in transcript[-6:]])
        outline_str = "\n".join([f"[{s.id}] {s.section}: " + ", ".join(q.question for q in s.questions) for s in outline])
        persona_str = f"Name: {persona.name}, Age: {persona.age}, Occupation: {persona.employment}, Personality: {persona.personality}"
        
        prompt = f"""
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
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            if isinstance(data, list):
                return [str(q) for q in data]
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return [str(q) for q in value]
            return []
        except Exception as e:
            print(f"Error generating questions from LLM: {e}")
            return [line.strip() for line in content.split("\n") if line.strip() and "Question" in line] if 'content' in locals() else []

@dataclass
class SuggestedQuestion:
    """Represents a single suggested question with its metadata."""
    question: str
    source: str  # 'dcagent_vX_AGENTNAME', 'llm'
    is_selected: bool

@dataclass
class SurveyItem:
    """Represents a single item in the survey data, containing context and suggested questions."""
    round_num: int
    previous_question: str
    answer: str
    suggested_questions: List[SuggestedQuestion]

class SurveyDataGenerator:
    """
    Generates survey data by processing a transcript and collecting question suggestions
    from both the DCAgentSystem and a direct LLM call.
    """
    def __init__(self, transcript_path: str, outline_path: str, persona_path: str, version: str):
        self.version = version
        DCAgentSystem, ConversationTurn, _, Persona, load_outline_from_sections = get_imports_for_version(version)
        self.ConversationTurn = ConversationTurn

        with open(transcript_path, "r", encoding="utf-8") as f:
            self.transcript_data = json.load(f)
        self.sections = load_outline_from_sections(outline_path)
        with open(persona_path, "r", encoding="utf-8") as f:
            self.persona = Persona(**json.load(f)["persona"])
        self.transcript = [self.ConversationTurn(**turn) for turn in self.transcript_data["transcript"]]

        self.dc_system = DCAgentSystem(
            sections=self.sections,
            persona=self.persona,
            session_id=f"survey_generator_session_{version}",
            user_preference="balanced",
        )
        self.llm_generator = LLMQuestionGenerator()
        
        self.survey_data: List[SurveyItem] = []

    async def generate_data(self):
        """
        Processes the transcript turn-by-turn to generate and collect questions.
        """
        print(f"\n=== Starting Survey Data Generation (Version: {self.version.upper()}) ===")
        for i in range(0, len(self.transcript) - 1, 2):
            prev_q_turn = self.transcript[i]
            prev_a_turn = self.transcript[i+1]
            round_num = i // 2
            print(f"\n[Round {round_num}] Processing...")

            transcript_so_far = self.transcript[:i + 2]

            all_suggestions: List[SuggestedQuestion] = []
            question_texts: Set[str] = set()

            selected_question_text: Optional[str] = None
            try:
                analysis_data, convergent_output = await self.dc_system.process_response(
                    prev_a_turn.text, transcript_so_far
                )
                selected_question_text = convergent_output.next_question

                # V3 has a different structure for analysis_data
                if self.version == 'v3':
                    dc_agent_questions = analysis_data.get("all_divergent_questions", [])
                else: # V1 and V2
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
                            source=f'dcagent_{self.version}_{div_q.source_agent_name}',
                            is_selected=is_champion
                        ))
                        question_texts.add(q_text)
                
                # If the selected champion from convergent_output was not in the divergent list, add it.
                if selected_question_text and not champion_found_in_list and selected_question_text not in question_texts:
                    all_suggestions.append(SuggestedQuestion(
                        question=selected_question_text,
                        source=f'dcagent_{self.version}_convergent_champion', # Add a specific source
                        is_selected=True
                    ))
                    question_texts.add(selected_question_text)

            except Exception as e:
                import traceback
                print(f"  ! Error getting questions from DCAgent: {e}")
                traceback.print_exc()

            llm_questions = await self.llm_generator.generate_questions(
                transcript_so_far, self.sections, self.persona, num_questions=4
            )
            for idx, q_text in enumerate(llm_questions):
                if q_text and q_text not in question_texts:
                    # The first question from the LLM is considered its "selected" choice.
                    is_llm_selected = (idx == 0)
                    all_suggestions.append(SuggestedQuestion(
                        question=q_text,
                        source='llm',
                        is_selected=is_llm_selected
                    ))
                    question_texts.add(q_text)

            self.survey_data.append(
                SurveyItem(
                    round_num=round_num,
                    previous_question=prev_q_turn.text,
                    answer=prev_a_turn.text,
                    suggested_questions=all_suggestions
                )
            )
            print(f"  > Generated {len(all_suggestions)} unique questions in total.")

    def save_data(self, output_path: str):
        """Saves the collected survey data to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        output_list = [asdict(item) for item in self.survey_data]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)
        print(f"\nSurvey data saved to {output_path}")

async def main():
    """Main function to run the survey data generation process."""
    parser = argparse.ArgumentParser(description='Generate survey data from an interview transcript.')
    parser.add_argument('--version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                        help="The version of the DCAgent system to use for question generation ('v1', 'v2', or 'v3')")
    parser.add_argument('--transcript-path', type=str, default='data/transcripts/mermaid_real.json',
                        help='Path to the input transcript JSON file.')
    parser.add_argument('--outline-path', type=str, default='data/outlines/example_outline.json',
                        help='Path to the outline JSON file.')
    parser.add_argument('--persona-path', type=str, default='data/personas/example_persona.json',
                        help='Path to the persona JSON file.')
    
    args = parser.parse_args()

    output_filename = f"comparison_data_{args.version}_{os.path.basename(args.transcript_path)}"
    
    generator = SurveyDataGenerator(
        transcript_path=args.transcript_path,
        outline_path=args.outline_path,
        persona_path=args.persona_path,
        version=args.version
    )
    await generator.generate_data()
    generator.save_data(f"data/evaluations/{output_filename}")

if __name__ == "__main__":
    asyncio.run(main()) 