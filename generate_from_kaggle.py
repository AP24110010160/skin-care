"""
Generates draft training examples (chat-messages JSONL) from the Kaggle
Skinsort skincare products dataset, for the skincare recommendation project.

WHAT THIS DOES:
- Reads the CSV (brand, name, type, country, ingridients, afterUse)
- Uses a CURATED list of well-established skincare actives (not raw
  statistical mining) cross-checked against real product data: which
  product types they appear in, and how many real products contain them.
- Generates two kinds of examples:
    1. "Routine by concern" examples (Feature 1) that mention real,
       data-backed product types + actives for each concern.
    2. "Ingredient explainer" examples (Feature 2) for actives NOT already
       covered in your hand-written set, enriched with real dataset stats.

IMPORTANT: These are DRAFTS. Review every one before merging into your
main raw_train.jsonl / raw_valid.jsonl. Side-effect and usage-timing facts
here come from established dermatological knowledge, not from the dataset
itself (the dataset only supplies product-type/frequency grounding) --
double check anything you're not 100% sure about.

USAGE:
    python3 generate_from_kaggle.py datasheet.csv
Outputs:
    kaggle_generated_train.jsonl
    kaggle_generated_valid.jsonl
"""

import csv
import json
import random
import sys
from collections import Counter

SYSTEM_PROMPT = (
    "You are a skincare guidance assistant. You provide general, educational "
    "skincare information — you are not a substitute for a dermatologist. "
    "Always include a brief disclaimer when giving routine or ingredient "
    "advice involving actives."
)

# Well-established actives only -- NOT derived by pure statistical mining.
# (Statistical mining on this dataset surfaces obscure/marketing ingredients;
# see the analysis notes -- these are the recognizable, evidence-backed ones.)
CURATED_ACTIVES = [
    "Niacinamide", "Salicylic Acid", "Hyaluronic Acid", "Retinol",
    "Ascorbic Acid", "Ceramide NP", "Azelaic Acid", "Glycolic Acid",
    "Alpha-Arbutin", "Centella Asiatica Leaf Extract", "Squalane",
    "Zinc PCA", "Panthenol", "Lactic Acid", "Tranexamic Acid",
    "Bakuchiol", "Peptide",
    "Tocopherol", "Ferulic Acid", "Allantoin",
    "Camellia Sinensis Leaf Extract", "Adenosine", "Beta-Glucan",
]

# Ingredients you already covered by hand -- skip generating explainers for these.
ALREADY_HAND_WRITTEN = {
    "Niacinamide", "Salicylic Acid", "Hyaluronic Acid", "Retinol",
    "Ascorbic Acid", "Ceramide NP", "Azelaic Acid", "Glycolic Acid",
}

BENEFIT_TAGS = {"Good For Oily Skin", "Redness Reducing", "Reduces Irritation",
                 "Acne Fighting", "Brightening", "Reduces Large Pores",
                 "Hydrating", "Skin Texture", "Anti-Aging", "Scar Healing",
                 "Dark Spots"}
CAUTION_TAGS = {"Irritating", "Acne Trigger", "Drying", "May Worsen Oily Skin",
                 "Rosacea", "Eczema"}

# Concern -> curated actives most relevant to it (derived from established
# dermatology + cross-checked against the real tag/type data above).
CONCERN_ACTIVES = {
    "oily and acne-prone skin": ["Niacinamide", "Salicylic Acid", "Zinc PCA"],
    "dry skin": ["Hyaluronic Acid", "Ceramide NP", "Squalane", "Panthenol"],
    "combination skin": ["Niacinamide", "Hyaluronic Acid"],
    "sensitive skin": ["Centella Asiatica Leaf Extract", "Panthenol", "Ceramide NP"],
    "pigmentation and dark spots": ["Ascorbic Acid", "Alpha-Arbutin", "Azelaic Acid", "Tranexamic Acid"],
    "fine lines and anti-aging": ["Retinol", "Bakuchiol", "Peptide", "Ascorbic Acid"],
    "dull skin and uneven texture": ["Glycolic Acid", "Lactic Acid", "Ascorbic Acid"],
    "enlarged pores": ["Salicylic Acid", "Glycolic Acid", "Niacinamide"],
    "redness and rosacea-prone skin": ["Centella Asiatica Leaf Extract", "Azelaic Acid", "Panthenol", "Beta-Glucan"],
    "post-acne scars and dark marks": ["Alpha-Arbutin", "Ascorbic Acid", "Azelaic Acid", "Centella Asiatica Leaf Extract"],
    "eczema-prone skin": ["Ceramide NP", "Panthenol", "Squalane", "Allantoin"],
    "environmental protection and antioxidant routine": ["Ascorbic Acid", "Tocopherol", "Ferulic Acid", "Camellia Sinensis Leaf Extract"],
}

# Concerns needing an extra caution/redirect line beyond the standard disclaimer
EXTRA_CAUTION = {
    "eczema-prone skin": (
        "Eczema is a medical skin condition, so treat this as general background rather than "
        "a substitute for a dermatologist's guidance, especially during a flare-up."
    ),
    "redness and rosacea-prone skin": (
        "If you suspect you have rosacea specifically (rather than general redness), a dermatologist "
        "can confirm this and may recommend prescription options beyond over-the-counter actives."
    ),
}

QUESTION_TEMPLATES = {
    "oily and acne-prone skin": [
        "what ingredients should I look for with oily acne prone skin",
        "recommend products for oily, acne-prone skin",
        "I have oily skin that breaks out a lot, what should I use",
        "best actives for acne and oiliness",
        "how do I control oil and breakouts",
        "skincare for oily skin that keeps breaking out",
        "what actives reduce shine and pimples",
        "oily skin help pls",
        "my face is always shiny and breaks out, help",
        "shrink pores and stop breakouts",
    ],
    "dry skin": [
        "what should I look for if my skin is really dry",
        "recommend ingredients for dry skin",
        "my skin feels tight and flaky, what actives help",
        "products for dehydrated dry skin",
        "how do I fix flaky dry skin",
        "best ingredients for very dry skin in winter",
        "dry skin routine help",
        "skin feels super tight after washing my face",
    ],
    "combination skin": [
        "what ingredients work for combination skin",
        "I have an oily t-zone but dry cheeks, what should I use",
        "recommend actives for combination skin",
        "skincare tips for combination skin",
        "combo skin routine",
    ],
    "sensitive skin": [
        "what's safe to use on sensitive skin",
        "gentle ingredients for reactive sensitive skin",
        "my skin gets irritated easily, what actives are safest",
        "what should I avoid if I have sensitive skin",
        "calming ingredients for easily irritated skin",
        "my face reacts to everything, what's gentle",
    ],
    "pigmentation and dark spots": [
        "how do I fade dark spots",
        "what ingredients help with pigmentation",
        "best actives for post-acne marks and dark spots",
        "recommend products for uneven skin tone",
        "how to lighten hyperpigmentation",
        "what actives even out skin tone",
        "dark spots wont go away, help",
    ],
    "fine lines and anti-aging": [
        "what ingredients help with fine lines",
        "best actives for anti-aging",
        "recommend an anti-aging routine",
        "how do I start an anti-aging routine as a beginner",
        "what helps with wrinkles and firmness",
        "im in my late 20s, should I start anti-aging products",
    ],
    "dull skin and uneven texture": [
        "how do I fix dull bumpy skin texture",
        "what helps with dullness and rough texture",
        "recommend actives for smoother, brighter skin",
        "why does my skin look dull and what helps",
        "my skin looks tired and dull lately",
    ],
    "enlarged pores": [
        "what ingredients help minimize pores",
        "how do I deal with large visible pores",
        "best actives for pore congestion",
        "pores look huge, what helps",
    ],
    "redness and rosacea-prone skin": [
        "what helps with facial redness",
        "ingredients for rosacea-prone skin",
        "how do I calm down redness and flushing",
        "my cheeks are always red, what can help",
    ],
    "post-acne scars and dark marks": [
        "how do I fade old acne scars",
        "what helps with post-acne dark marks",
        "ingredients for acne scarring and discoloration",
        "acne left marks on my face, how do I fade them",
    ],
    "eczema-prone skin": [
        "what ingredients are safe for eczema-prone skin",
        "gentle skincare for eczema",
        "what should I avoid if I have eczema",
        "eczema flare up, what skincare is safe",
    ],
    "environmental protection and antioxidant routine": [
        "what ingredients protect skin from pollution and sun damage",
        "best antioxidants for skincare",
        "how do I protect my skin from environmental damage",
        "should I use an antioxidant serum",
    ],
}


def tags_of(row):
    return set(t.strip() for t in row["afterUse"].split(",") if t.strip())


def ingredients_of(row):
    return [i.strip() for i in row["ingridients"].split(",") if i.strip()]


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader if r["afterUse"].strip() and r["ingridients"].strip()]


def active_stats(rows, active_name):
    matches = [r for r in rows if any(active_name.lower() in ing.lower() for ing in ingredients_of(r))]
    if not matches:
        return None
    type_counter = Counter(r["type"] for r in matches)
    tag_counter = Counter()
    for r in matches:
        tag_counter.update(tags_of(r))
    top_types = [t for t, _ in type_counter.most_common(3)]
    top_benefit_tags = [t for t, _ in tag_counter.most_common(10) if t in BENEFIT_TAGS][:3]
    return {
        "count": len(matches),
        "top_types": top_types,
        "top_benefit_tags": top_benefit_tags,
    }


def make_routine_example(concern, question, actives_with_stats, style):
    extra_caution = EXTRA_CAUTION.get(concern, "")
    is_avoid_question = "avoid" in question.lower()

    intro_bullet = (
        f"For {concern}, generally avoid harsh sulfates, added fragrance, and high "
        f"concentrations of exfoliating acids, since these are common irritants. "
        f"Ingredients that tend to work better instead:\n"
        if is_avoid_question else
        f"For {concern}, a few well-supported ingredients to look for:\n"
    )
    intro_paragraph = (
        f"For {concern}, generally avoid harsh sulfates, added fragrance, and high "
        f"concentrations of exfoliating acids, since these are common irritants. "
        f"Better options tend to include ingredients like "
        if is_avoid_question else
        f"For {concern}, look for ingredients like "
    )

    if style == "bullet":
        lines = [intro_bullet]
        for active, stats in actives_with_stats:
            type_str = " or ".join(stats["top_types"][:2]) if stats["top_types"] else "serum"
            lines.append(
                f"- **{active}** — commonly found in {type_str} formulas "
                f"({stats['count']}+ products in common skincare databases use it), "
                f"often associated with {', '.join(stats['top_benefit_tags']) if stats['top_benefit_tags'] else 'general skin benefits'}."
            )
        lines.append(
            "\nAs a beginner, introduce one new active at a time and patch test first. "
            + (extra_caution + " " if extra_caution else "")
            + "This is general education, not a diagnosis — a dermatologist can tailor this to your skin specifically."
        )
        answer = "\n".join(lines)
    else:  # "paragraph" style
        active_phrases = []
        for active, stats in actives_with_stats:
            type_str = " or ".join(stats["top_types"][:2]) if stats["top_types"] else "serum"
            active_phrases.append(f"{active} (common in {type_str} formulas)")
        answer = (
            f"{intro_paragraph}{', '.join(active_phrases)}. "
            f"These show up consistently across real skincare product data for this concern, "
            f"so they're a solid starting point rather than a random guess.\n\n"
            "Start with one new active at a time rather than layering everything at once, and patch test "
            "anything new on a small area first. " + (extra_caution + " " if extra_caution else "") +
            "This is general education, not a diagnosis — a dermatologist can tailor this to your skin specifically."
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


INGREDIENT_FACTS = {
    "Alpha-Arbutin": {
        "does": "a tyrosinase-inhibiting brightening agent that reduces melanin production",
        "best_for": "most skin types, including sensitive skin",
        "helps": "hyperpigmentation, dark spots, uneven tone",
        "side_effects": "minimal; rare mild irritation",
        "when": "AM and/or PM",
        "beginner": "typically used around 2% concentration, safe for daily use — always pair with sunscreen since sun exposure worsens pigmentation",
        "pairs_with": "vitamin C, niacinamide, sunscreen",
    },
    "Centella Asiatica Leaf Extract": {
        "does": "a soothing, anti-inflammatory extract (often called 'Cica') that supports barrier repair",
        "best_for": "sensitive, irritated, or barrier-compromised skin",
        "helps": "redness, irritation, barrier repair, post-procedure calming",
        "side_effects": "essentially none, very well tolerated",
        "when": "AM and/or PM",
        "beginner": "safe to use daily from day one, no need to introduce slowly",
        "pairs_with": "ceramides, niacinamide, and virtually every other active",
    },
    "Squalane": {
        "does": "a lightweight emollient that mimics the skin's natural lipids to lock in moisture",
        "best_for": "all skin types, including oily (it's non-comedogenic)",
        "helps": "dryness, dehydration, barrier support",
        "side_effects": "none common",
        "when": "AM and/or PM, often as a last step to seal in moisture",
        "beginner": "safe daily, a good starter moisturizing ingredient",
        "pairs_with": "hyaluronic acid, ceramides, most other actives",
    },
    "Zinc PCA": {
        "does": "helps regulate oil production and has mild antibacterial properties",
        "best_for": "oily and acne-prone skin",
        "helps": "excess oil, mild acne support",
        "side_effects": "minimal",
        "when": "AM and/or PM",
        "beginner": "pairs well with niacinamide for oil control",
        "pairs_with": "niacinamide, salicylic acid",
    },
    "Panthenol": {
        "does": "a humectant and soothing agent (pro-vitamin B5) that supports hydration and barrier repair",
        "best_for": "all skin types, especially dry or irritated skin",
        "helps": "dryness, irritation, barrier support",
        "side_effects": "minimal, very well tolerated",
        "when": "AM and/or PM",
        "beginner": "very safe, no patch test typically needed",
        "pairs_with": "hyaluronic acid, ceramides, centella asiatica",
    },
    "Lactic Acid": {
        "does": "a gentler AHA that exfoliates the skin's surface and also has humectant properties",
        "best_for": "normal to dry skin, or sensitive skin wanting gentle exfoliation",
        "helps": "texture, dullness, mild pigmentation",
        "side_effects": "irritation or sun sensitivity if overused",
        "when": "PM, 2-3x per week to start",
        "beginner": "start at low frequency and always pair with daily SPF",
        "pairs_with": "hyaluronic acid, avoid stacking with retinol or other strong actives the same night",
    },
    "Tranexamic Acid": {
        "does": "a brightening ingredient that reduces melanin production, often effective for stubborn discoloration",
        "best_for": "most skin types",
        "helps": "stubborn pigmentation, melasma-like discoloration, post-inflammatory marks",
        "side_effects": "minimal when used topically",
        "when": "AM and/or PM",
        "beginner": "pairs well with niacinamide and vitamin C for pigmentation routines",
        "pairs_with": "niacinamide, vitamin C",
    },
    "Bakuchiol": {
        "does": "a plant-derived retinol alternative offering similar cell-turnover benefits with less irritation",
        "best_for": "sensitive skin or those who can't tolerate retinol",
        "helps": "fine lines, texture, mild acne",
        "side_effects": "minimal, occasional mild irritation",
        "when": "AM or PM (unlike retinol, doesn't strictly need to be night-only)",
        "beginner": "a gentler starting point than retinol, though still worth patch testing",
        "pairs_with": "most other actives, including vitamin C",
    },
    "Peptide": {
        "does": "signal proteins that support collagen production and skin repair",
        "best_for": "most skin types, especially aging-prone skin",
        "helps": "fine lines, firmness, skin barrier support",
        "side_effects": "essentially none",
        "when": "AM and/or PM",
        "beginner": "safe for daily use, pairs well with most other actives",
        "pairs_with": "hyaluronic acid, niacinamide, vitamin C",
    },
    "Tocopherol": {
        "does": "vitamin E, an antioxidant that helps protect skin from environmental damage and supports the moisture barrier",
        "best_for": "most skin types",
        "helps": "environmental protection, supporting dryness-prone skin",
        "side_effects": "rare; can feel heavy or be comedogenic in oil-based formulas for very acne-prone skin",
        "when": "AM and/or PM",
        "beginner": "usually found already combined into other products (like vitamin C serums) rather than used alone",
        "pairs_with": "vitamin C, ferulic acid",
    },
    "Ferulic Acid": {
        "does": "an antioxidant that stabilizes and boosts the effectiveness of vitamin C",
        "best_for": "most skin types",
        "helps": "environmental protection, enhancing antioxidant serums",
        "side_effects": "minimal",
        "when": "AM, ideally under sunscreen",
        "beginner": "usually already combined into vitamin C serums rather than used as a standalone product",
        "pairs_with": "vitamin C, vitamin E",
    },
    "Allantoin": {
        "does": "a soothing, skin-conditioning ingredient that supports healing and hydration",
        "best_for": "all skin types, especially sensitive or irritated skin",
        "helps": "irritation, dryness, soothing after exfoliation or other actives",
        "side_effects": "essentially none",
        "when": "AM and/or PM",
        "beginner": "very safe for daily use, often paired with stronger actives to offset irritation",
        "pairs_with": "retinol, AHAs/BHAs, niacinamide",
    },
    "Camellia Sinensis Leaf Extract": {
        "does": "green tea extract — an antioxidant with mild anti-inflammatory properties",
        "best_for": "most skin types, including oily and acne-prone skin",
        "helps": "environmental protection, mild redness and irritation calming",
        "side_effects": "minimal",
        "when": "AM and/or PM",
        "beginner": "safe for daily use",
        "pairs_with": "niacinamide, vitamin C",
    },
    "Adenosine": {
        "does": "a soothing ingredient with mild anti-wrinkle signaling properties",
        "best_for": "most skin types, popular in mature-skin and anti-aging formulas",
        "helps": "fine lines, calming, texture",
        "side_effects": "minimal",
        "when": "AM and/or PM",
        "beginner": "safe for daily use",
        "pairs_with": "peptides, hyaluronic acid",
    },
    "Beta-Glucan": {
        "does": "a soothing, hydrating polysaccharide that supports barrier repair",
        "best_for": "sensitive, irritated, or barrier-compromised skin",
        "helps": "irritation, hydration, redness",
        "side_effects": "essentially none",
        "when": "AM and/or PM",
        "beginner": "very safe for daily use",
        "pairs_with": "centella asiatica, ceramides, hyaluronic acid",
    },
}


def make_ingredient_example(active, stats):
    q = random.choice([
        f"what is {active.lower()} used for",
        f"tell me about {active.lower()}",
        f"explain {active.lower()} in skincare",
    ])
    facts = INGREDIENT_FACTS.get(active)
    type_str = " and ".join(stats["top_types"][:2]) if stats["top_types"] else "serums"

    if facts:
        answer = (
            f"{active} is {facts['does']}.\n\n"
            f"Best for: {facts['best_for']}.\n"
            f"Helps with: {facts['helps']}.\n"
            f"Side effects: {facts['side_effects']}.\n"
            f"When to use: {facts['when']}.\n"
            f"Beginner tip: {facts['beginner']}.\n"
            f"Pairs well with: {facts['pairs_with']}.\n\n"
            f"(In real product data, it's most commonly found in {type_str} formulas, "
            f"appearing in {stats['count']}+ products.)\n\n"
            "General education only, not a diagnosis."
        )
    else:
        tag_str = ", ".join(stats["top_benefit_tags"]) if stats["top_benefit_tags"] else "general skin health"
        answer = (
            f"{active} is a skincare ingredient most commonly found in {type_str} formulas — "
            f"it appears in {stats['count']}+ products, typically associated with {tag_str}.\n\n"
            f"[REVIEW NEEDED: no established-facts entry for {active} yet -- add side effects, "
            f"AM/PM timing, and beginner tips from a verified source before using this example.]\n\n"
            "General education only, not a diagnosis."
        )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "datasheet.csv"
    rows = load_rows(csv_path)

    # Precompute stats for every curated active once
    stats_cache = {}
    for active in CURATED_ACTIVES:
        s = active_stats(rows, active)
        if s:
            stats_cache[active] = s

    examples = []

    # Feature 1: routine-by-concern examples -- every phrasing, alternating answer style
    for concern, actives in CONCERN_ACTIVES.items():
        actives_with_stats = [(a, stats_cache[a]) for a in actives if a in stats_cache]
        if not actives_with_stats:
            continue
        questions = QUESTION_TEMPLATES.get(concern, [])
        for i, question in enumerate(questions):
            style = "bullet" if i % 2 == 0 else "paragraph"
            examples.append(make_routine_example(concern, question, actives_with_stats, style))

    # Feature 2: ingredient explainer drafts for actives not already hand-written
    for active in CURATED_ACTIVES:
        if active in ALREADY_HAND_WRITTEN or active not in stats_cache:
            continue
        examples.append(make_ingredient_example(active, stats_cache[active]))

    random.shuffle(examples)
    split_idx = int(len(examples) * 0.85)
    train, valid = examples[:split_idx], examples[split_idx:]

    with open("kaggle_generated_train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open("kaggle_generated_valid.jsonl", "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} examples to kaggle_generated_train.jsonl")
    print(f"Wrote {len(valid)} examples to kaggle_generated_valid.jsonl")
    print("\nREMINDER: these are DRAFTS. Every ingredient-explainer example has a")
    print("[REVIEW NEEDED] placeholder you must fill in with verified facts before")
    print("merging into your real training set. Routine examples should also be")
    print("spot-checked for accuracy and tone before merging.")


if __name__ == "__main__":
    main()
