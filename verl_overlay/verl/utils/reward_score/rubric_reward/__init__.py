"""Rubric-based reward for RL in non-verifiable domains.

``rubric_generator``  -- stages 1-4: pseudo-reference, near-miss,
                        contrastive criteria, rubric.
``rurbichub_v1_Medical`` -- stage 5: the judge, the scorer, and the
                        ``compute_score`` entry point verl calls.
"""
