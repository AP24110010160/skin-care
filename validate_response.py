"""
Validates a model-generated skincare answer against the verified knowledge
base (skincare_kb.json), specifically to catch two failure modes observed
during testing:

1. INGREDIENT SUBSTITUTION: the user asks about ingredients A and B, but the
   model's answer is actually about A and C (a similar-sounding pair from
   training). This is exactly the bug found when testing "glycolic acid and
   retinol" -> got an answer about "salicylic acid and glycolic acid" instead.

2. RELATIONSHIP CONTRADICTION: the model's answer claims a relationship
   (safe / avoid / alternate) that contradicts the verified knowledge base
   for that exact pair.

USAGE:
    from validate_response import validate_response
    result = validate_response(user_question, model_answer)
    if not result["ok"]:
        print(result["warnings"])
"""

import json
import re

with open("skincare_kb.json") as f:
    KB = json.load(f)

INGREDIENTS = KB["ingredients"]
CONFLICT_CLASSES = KB["conflict_classes"]
EXPLICIT_PAIRS = KB["explicit_pairs"]

# Build alias -> canonical name lookup, longest alias first so e.g.
# "vitamin c" matches before a shorter, less specific alias would.
_ALIAS_TO_NAME = []
for name, info in INGREDIENTS.items():
    for alias in info["aliases"]:
        _ALIAS_TO_NAME.append((alias.lower(), name))
_ALIAS_TO_NAME.sort(key=lambda x: -len(x[0]))

# Build explicit pair lookup, keyed by frozenset so order doesn't matter
_PAIR_LOOKUP = {}
for entry in EXPLICIT_PAIRS:
    key = frozenset(entry["pair"])
    _PAIR_LOOKUP[key] = entry


def extract_ingredients(text):
    """Return the set of canonical ingredient names mentioned in text."""
    text_lower = text.lower()
    found = set()
    for alias, name in _ALIAS_TO_NAME:
        if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
            found.add(name)
    return found


def get_relationship(name_a, name_b):
    """
    Look up the relationship between two canonical ingredient names.
    Checks explicit pairs first, then falls back to conflict-class defaults.
    Returns a dict with 'relationship' and 'note'.
    """
    key = frozenset([name_a, name_b])
    if key in _PAIR_LOOKUP:
        entry = _PAIR_LOOKUP[key]
        return {"relationship": entry["relationship"], "note": entry["note"]}

    class_a = INGREDIENTS.get(name_a, {}).get("conflict_class")
    class_b = INGREDIENTS.get(name_b, {}).get("conflict_class")

    if class_a == "gentle_universal" or class_b == "gentle_universal":
        return {
            "relationship": "safe_together",
            "note": f"{name_a if class_a == 'gentle_universal' else name_b} "
                    f"has no known conflicts, so this pairing is generally safe.",
        }
    if class_a == "strong_exfoliant_night" and class_b == "strong_exfoliant_night":
        return {
            "relationship": "alternate_nights",
            "note": "Both are strong actives typically used at night; "
                     "alternating nights is the generally recommended default.",
        }
    if {class_a, class_b} == {"strong_exfoliant_night", "ph_sensitive_am"}:
        return {
            "relationship": "sequential",
            "note": "One works best in the AM, the other is a strong PM active; "
                     "using them at different times is the generally recommended default.",
        }
    return {"relationship": "unknown", "note": "No specific rule on file for this pair -- verify manually."}


# Words that signal what relationship the ANSWER TEXT is claiming,
# used to sanity-check the model's stated advice against the KB.
_CLAIM_KEYWORDS = {
    "avoid_or_alternate": ["not recommended", "avoid", "alternate night", "different night", "don't combine", "do not combine"],
    "safe": ["safe to use together", "generally yes", "yes, this is a common", "commonly combined", "pairs well", "can be used together"],
    "sequential": ["am and pm", "am, then", "in the morning", "at night instead"],
}


def _claimed_relationship(answer_text):
    text = answer_text.lower()
    for label, keywords in _CLAIM_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return label
    return None


def validate_response(question, answer):
    """
    Main entry point. Returns:
    {
        "ok": bool,
        "question_ingredients": set,
        "answer_ingredients": set,
        "warnings": [str, ...],
    }
    """
    warnings = []
    q_ings = extract_ingredients(question)
    a_ings = extract_ingredients(answer)

    # Only run pair-specific checks when the question is clearly about
    # exactly two known ingredients (the common "can I mix X and Y" case).
    if len(q_ings) == 2:
        if not q_ings.issubset(a_ings):
            missing = q_ings - a_ings
            warnings.append(
                f"INGREDIENT MISMATCH: question asked about {sorted(q_ings)}, "
                f"but the answer doesn't clearly mention {sorted(missing)}. "
                f"The model may have substituted a different, similar-sounding pair."
            )
        else:
            name_a, name_b = sorted(q_ings)
            expected = get_relationship(name_a, name_b)
            claimed = _claimed_relationship(answer)
            if expected["relationship"] != "unknown" and claimed:
                mismatch = (
                    (expected["relationship"] == "safe_together" and claimed == "avoid_or_alternate")
                    or (expected["relationship"] in ("alternate_nights", "avoid") and claimed == "safe")
                )
                if mismatch:
                    warnings.append(
                        f"RELATIONSHIP MISMATCH for {name_a} + {name_b}: "
                        f"verified guidance is '{expected['relationship']}' ({expected['note']}), "
                        f"but the answer seems to claim '{claimed}'."
                    )

    return {
        "ok": len(warnings) == 0,
        "question_ingredients": q_ings,
        "answer_ingredients": a_ings,
        "warnings": warnings,
    }


if __name__ == "__main__":
    # Worked example: reproduces the exact bug found during testing.
    question = "can I mix glycolic acid and retinol"
    bad_answer = (
        "Not recommended on the same night, especially as a beginner — both are "
        "strong actives with different mechanisms of action, and using them "
        "together in the same step can cause irritation or worsen dryness.\n\n"
        "A gentler approach: use salicylic acid in the AM to start, then mix "
        "glycolic acid in the PM for exfoliation."
    )
    result = validate_response(question, bad_answer)
    print("Question:", question)
    print("Answer ingredients found:", result["answer_ingredients"])
    print("Question ingredients found:", result["question_ingredients"])
    print("OK?", result["ok"])
    for w in result["warnings"]:
        print(" -", w)

    print()

    # A correct answer for comparison
    good_answer = (
        "Not recommended on the same night — both increase cell turnover and "
        "sensitivity. Alternate nights: glycolic acid Mon/Wed/Fri, retinol Tue/Thu/Sat."
    )
    result2 = validate_response(question, good_answer)
    print("Question:", question)
    print("Answer ingredients found:", result2["answer_ingredients"])
    print("OK?", result2["ok"])
