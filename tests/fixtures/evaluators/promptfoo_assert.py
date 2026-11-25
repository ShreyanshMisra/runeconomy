"""A Promptfoo python assertion file, written exactly as for Promptfoo.

Promptfoo config: `assert: [{type: python, value: file://promptfoo_assert.py}]`
"""


def get_assert(output: str, context) -> dict:
    expected = context["vars"]["expected"]
    passed = output.strip().lower() == expected.lower()
    return {"pass": passed, "score": 1.0 if passed else 0.0, "reason": "exact answer" if passed else "wrong answer"}
