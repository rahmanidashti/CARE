[System]
You are an expert answer writer. Given a strong correct answer, you produce a subtly incorrect variant -- one that looks plausible but fails on one or two specific reasoning steps. The wrong answer should be close to the correct one, not obviously bad.

[User]
You are given a QUESTION and a CORRECT ANSWER.

Write a BORDERLINE-WRONG answer: a response that looks plausible and is similar
in style to the correct answer, but fails on one or two specific reasoning steps.

Rules:
- Keep it close to the correct answer -- same general approach, similar length.
- Introduce a subtle error (e.g. a wrong conclusion, a flawed reasoning step, a
  missing key condition) -- not an obvious or absurd mistake.
- Do NOT make it completely wrong or radically different.

Return ONLY the wrong answer text (no preamble, no labels, no JSON).

QUESTION:
"""{question}"""

CORRECT ANSWER:
"""{pseudo_reference}"""
