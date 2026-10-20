"""Golden recipes: the code that turns library outputs into flat golden values.

The recipes import nothing of ``dof``. The same recipe runs on the reference implementation (when a golden is
recorded) and on ``dof`` (when it is tested), each through a backend that only routes calls. Every golden file
records ``recipe_sha256``; a changed recipe makes the golden tests stop until the goldens are recorded again.
"""
