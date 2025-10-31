#!/usr/bin/env python3
"""
Solution for LLM Hallucination Robustness Challenge
This solution uses a quantized Russian LLM with anti-hallucination strategies
"""

import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
import re


class HallucinationResistantLLM:
    def __init__(self):
        """Initialize the model with anti-hallucination configuration"""

        # Model selection - using a capable Russian model
        # Saiga models are fine-tuned for Russian and follow instructions well
        self.model_name = "IlyaGusev/saiga_mistral_7b_lora"

        print(f"Loading model: {self.model_name}")

        # Quantization config for faster inference and smaller memory footprint
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        self.model.eval()

        # System prompt emphasizing honesty and refusing to answer when uncertain
        self.system_prompt = """Ты честный и точный ассистент. Отвечай ТОЛЬКО если абсолютно уверен в правильности ответа.

СТРОГИЕ ПРАВИЛА:
1. Если не уверен на 100% - отвечай "Я не знаю"
2. Если вопрос содержит невозможное или ложную предпосылку - отвечай "Я не могу ответить на этот вопрос"
3. Если вопрос абсурдный или внутренне противоречивый - отвечай "Я не могу ответить на этот вопрос"
4. Давай КРАТКИЙ ответ, только факт, без пояснений
5. НИКОГДА не выдумывай: факты, имена, названия книг, фильмов, мест, дат, событий
6. При малейшем сомнении - говори "Я не знаю"

ПРИМЕРЫ НЕВОЗМОЖНЫХ ВОПРОСОВ (отвечай отказом):
- Анахронизмы: "Какой античный математик изобрёл компьютер?"
- Физически невозможное: "На какой планете живут единороги?"
- Несуществующее: "Кто написал книгу, которая не существует?"
- Ложные предпосылки: "Какого цвета прозрачная вода?"

Отвечай ТОЛЬКО на вопрос, без "Ответ:", без пояснений."""

    def detect_hallucination_provocation(self, question: str) -> bool:
        """
        Detect if a question is likely a hallucination provocation
        Returns True if the question seems designed to elicit hallucination
        """
        question_lower = question.lower()

        # Patterns that suggest hallucination provocations
        provocation_patterns = [
            # Temporal anachronisms - ancient + modern tech
            r'(античн|древн|первобытн|средневеков).*(компьютер|интернет|телефон|самолёт|двигател|ракет|электричеств)',
            r'(до нашей эры|до н\.э\.|до христ).*(компьютер|интернет|телефон|самолёт|двигател)',

            # Impossible combinations
            r'подводн.*(космическ|лун|марсианск)',
            r'космическ.*(подводн|океан)',

            # Fictional entities treated as real
            r'(единорог|дракон|эльф|гном|хоббит|зомби|вампир).*(жив|обитал|населя|страна)',

            # Time travel paradoxes
            r'будущ.*(изобр.*прошл|создал.*до)',

            # Physically impossible
            r'какого цвета.*(прозрачн|невидим)',
            r'сколько весит.*(невесом|пустот)',

            # Non-existent but plausible-sounding things
            r'несуществующ.*(книг|фильм|человек|компани)',
        ]

        for pattern in provocation_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def generate_answer(self, question: str) -> str:
        """
        Generate an answer for a single question with hallucination resistance
        """
        # Check for obvious provocation
        if self.detect_hallucination_provocation(question):
            return "Я не могу ответить на этот вопрос"

        # Construct the conversation
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        # Format for Saiga model
        prompt = self.format_prompt(conversation)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate with conservative parameters to reduce hallucinations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Short answers
                temperature=0.3,  # Low temperature for consistency
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Extract the generated text
        generated_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Clean up the answer
        answer = self.clean_answer(answer)

        # Validate the answer format
        if not self.validate_answer(answer, question):
            return "Я не знаю"

        # Calculate confidence based on generation scores
        confidence = self.calculate_confidence(outputs.scores)

        # If confidence is too low, say "I don't know"
        if confidence < 0.4:
            return "Я не знаю"

        # Check if the answer itself indicates uncertainty
        if self.is_uncertain_answer(answer):
            return "Я не знаю"

        return answer

    def format_prompt(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for Saiga model"""
        prompt = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"<s>system\n{content}</s>\n"
            elif role == "user":
                prompt += f"<s>user\n{content}</s>\n"
            elif role == "assistant":
                prompt += f"<s>assistant\n{content}</s>\n"
        prompt += "<s>assistant\n"
        return prompt

    def clean_answer(self, answer: str) -> str:
        """Clean up the generated answer"""
        # Remove common artifacts
        answer = answer.split("</s>")[0]
        answer = answer.split("<s>")[0]
        answer = answer.strip()

        # Remove leading "Ответ:" if present
        answer = re.sub(r'^(Ответ|Answer):\s*', '', answer, flags=re.IGNORECASE)

        return answer

    def calculate_confidence(self, scores) -> float:
        """
        Calculate confidence score from generation scores
        Higher confidence means the model is more certain
        """
        if scores is None or len(scores) == 0:
            return 0.5

        # Calculate average probability of top token at each step
        confidences = []
        for score in scores[:10]:  # Look at first 10 tokens
            probs = torch.softmax(score[0], dim=-1)
            max_prob = torch.max(probs).item()
            confidences.append(max_prob)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def is_uncertain_answer(self, answer: str) -> bool:
        """Check if the answer itself expresses uncertainty"""
        uncertain_phrases = [
            'не уверен',
            'не могу сказать',
            'точно не знаю',
            'вероятно',
            'возможно',
            'может быть',
            'не могу ответить',
            'затрудняюсь',
            'не имею информации',
            'мне неизвестно',
            'недостаточно информации',
            'не располагаю',
            'сложно сказать',
            'трудно сказать',
            'предположительно',
            'скорее всего',
            'по-видимому',
        ]

        answer_lower = answer.lower()

        # Check for uncertainty phrases
        if any(phrase in answer_lower for phrase in uncertain_phrases):
            return True

        # Check if answer is suspiciously vague or hedging
        hedge_words = ['вероятно', 'возможно', 'может', 'наверное', 'кажется', 'думаю', 'полагаю']
        hedge_count = sum(1 for word in hedge_words if word in answer_lower)
        if hedge_count >= 2:
            return True

        return False

    def validate_answer(self, answer: str, question: str) -> bool:
        """
        Validate if the answer looks reasonable
        Returns False if answer seems like hallucination
        """
        # Empty or very short answers are suspicious
        if len(answer.strip()) < 2:
            return False

        # Extremely long answers might be rambling/hallucinating
        if len(answer.split()) > 100:
            return False

        # Check for model artifacts or meta-commentary
        meta_phrases = [
            'как языковая модель',
            'как ии',
            'я не могу',
            'извините',
            'к сожалению',
        ]
        answer_lower = answer.lower()
        for phrase in meta_phrases:
            if phrase in answer_lower:
                # These are often signs the model is refusing properly
                return True

        return True

    def process_batch(self, questions: List[str]) -> List[str]:
        """Process a batch of questions"""
        answers = []
        for i, question in enumerate(questions):
            if i % 100 == 0:
                print(f"Processing question {i}/{len(questions)}")

            try:
                answer = self.generate_answer(question)
                answers.append(answer)
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                # On error, default to "I don't know" to avoid hallucinations
                answers.append("Я не знаю")

        return answers


def main():
    """Main entry point"""
    # Read input from stdin or file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    # Extract questions
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict) and 'questions' in data:
        questions = data['questions']
    else:
        raise ValueError("Unexpected input format")

    print(f"Loaded {len(questions)} questions")

    # Initialize model
    llm = HallucinationResistantLLM()

    # Process questions
    answers = llm.process_batch(questions)

    # Output results
    result = {"answers": answers}

    # Write to stdout or file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
