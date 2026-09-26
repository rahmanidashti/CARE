You are an expert evaluator designing a grading rubric. Given a user prompt and an
ideal reference answer, produce a list of about {n_items} rubric criteria that a high-quality
response should satisfy.

Requirements:
- Each criterion must be a single, atomic, objectively checkable statement (PRESENT / NOT_PRESENT).
- Assign each criterion a positive integer "points" value reflecting its importance.
- Base the criteria on the substance of the reference answer, but phrase them so they can be
  checked against ANY response (do not reference "the reference answer" in the text).
- Cover correctness, completeness, safety, and clarity as relevant to the prompt.

[Optional contrastive-guidance block: inserted only when example criteria are available]
For inspiration, here are a few example criteria that capture key distinctions for this question. You are NOT required to use them -- treat them as a hint about what matters, then write the best rubric you can from the reference answer itself:
  Example 1: {criterion_1}
  ...
  Example {m}: {criterion_m}

Start your response with a valid JSON array that starts with "```json" and ends with "```". Each element must be an object with keys "criterion" (string) and "points" (number). Do not include any extra text or explanations.

Example response:
```json
[
 {"criterion": "States that the condition requires urgent medical attention", "points": 5},
 {"criterion": "Mentions at least two common symptoms", "points": 3}
]
```

<Prompt>
{prompt}
</Prompt>

<ReferenceAnswer>
{pseudo_reference}
</ReferenceAnswer>
