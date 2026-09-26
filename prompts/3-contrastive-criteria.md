[System]
You are an expert evaluator. Given a correct answer and a borderline-wrong variant, you identify the specific criteria that separate them -- the minimal distinctions that make the correct answer right and the wrong answer fail.

[User]
You are given a QUESTION, a CORRECT ANSWER, and a BORDERLINE-WRONG answer.

Identify exactly {num_examples} rubric criteria that separate the CORRECT answer
from the BORDERLINE-WRONG answer -- criteria the correct answer satisfies but the
wrong answer fails.

Each criterion MUST have three components:
- "criterion": a clear, self-contained positive statement a good answer satisfies.
- "weight": a number in [0, 1] (weights should sum to roughly 1).
- "category": a short label, e.g. "correctness", "reasoning", "completeness".

Keep criteria specific to THIS question. Phrase so that "satisfied" = good.

Return ONLY valid JSON:
{"rubrics": [{"criterion": "...", "weight": 0.0, "category": "..."}]}

QUESTION:
"""{question}"""

CORRECT ANSWER:
"""{pseudo_reference}"""

BORDERLINE-WRONG ANSWER:
"""{boundary}"""
