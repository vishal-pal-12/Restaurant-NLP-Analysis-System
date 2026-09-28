"""
Experiment 8: Named Entity Recognition (NER)

Extracts domain-specific and general named entities:
- RESTAURANT
- FOOD
- LOCATION
- PERSON
- DATE
- ORGANIZATION

Combines NLTK's MaxEnt ne_chunk with a comprehensive culinary gazetteer and regex patterns.
"""

import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk import ne_chunk, pos_tag, Tree
from nltk.tokenize import word_tokenize


# Culinary Gazetteer & Patterns
KNOWN_FOOD_ITEMS = {
    "pizza", "margherita pizza", "pepperoni pizza", "pasta", "pasta carbonara", "alfredo pasta",
    "biryani", "dum biryani", "chicken biryani", "mutton biryani", "hyderabadi biryani",
    "burger", "craft burger", "cheeseburger", "fries", "french fries", "peri peri fries",
    "paneer tikka", "dal makhani", "butter chicken", "garlic naan", "roti", "parotta",
    "dosa", "masala dosa", "mysore dosa", "idli", "vada", "sambar", "filter coffee",
    "kebab", "seekh kebab", "galouti kebab", "tandoori chicken", "mutton rogan josh",
    "noodles", "hakka noodles", "fried rice", "dim sum", "dumplings", "manchurian",
    "sushi", "ramen", "bao", "pork bao", "thai green curry", "pad thai",
    "ice cream", "chocolate lava cake", "brownie", "sizzling brownie", "waffle", "nutella waffle",
    "cheesecake", "tiramisu", "gulab jamun", "jalebi", "prawn curry", "fish fry", "surmai",
    "mocktail", "cocktail", "cold coffee", "lemonade", "bun maska", "irani chai", "kheema pav"
}

KNOWN_LOCATIONS = {
    "pune", "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
    "fc road", "viman nagar", "kothrud", "koregaon park", "bandra", "andheri", "powai",
    "indiranagar", "connaught place", "baner", "aundh", "camp", "hinjewadi", "shivaji nagar"
}

KNOWN_RESTAURANTS = {
    "tasty bites", "punjab grill", "punjab grill junction", "spice symphony", "cafe goodluck",
    "little italy", "biryani by kilo", "vaishali", "vaishali restaurant", "mainland china",
    "dario's trattoria", "the belgian waffle co", "barbeque nation", "malaka spice",
    "shabree", "shabree restaurant", "chili's", "chili's american grill", "sujata mastani",
    "the bombay canteen", "dakshin", "bukhara", "toit brewpub", "german bakery",
    "copper chimney", "saravanaa bhavan", "wok express", "farzi cafe", "saby's pizzeria",
    "oh! calcutta", "karim's", "the fatty bao", "george restaurant", "keventers", "kareem's",
    "le plaisir", "empire restaurant", "bheema's", "blue nile", "abc restaurant"
}

KNOWN_ORGANIZATIONS = {
    "zomato", "swiggy", "ubereats", "doordash", "deliveroo", "mcdonald's", "kfc", "domino's",
    "starbucks", "subway", "pizza hut", "burger king", "marriott", "taj hotels", "hyatt"
}

KNOWN_DATES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "yesterday", "tomorrow", "tonight", "last night", "last week", "weekend",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"
}

ENTITY_COLORS = {
    "RESTAURANT": "#FF4B4B",
    "FOOD": "#FFA421",
    "LOCATION": "#1E88E5",
    "PERSON": "#9C27B0",
    "DATE": "#00897B",
    "ORGANIZATION": "#5E35B1"
}


def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extracts named entities from restaurant reviews and food queries using a hybrid approach:
    1. Gazetteer matching for Food, Location, and Restaurant phrases.
    2. Date & Temporal expressions.
    3. NLTK ne_chunk for Person, Organization, and GPE locations.
    """
    entities = []
    text_lower = text.lower()
    matched_spans = set()

    # 1. Match Known Restaurants
    for r in sorted(KNOWN_RESTAURANTS, key=len, reverse=True):
        pattern = r"\b" + re.escape(r) + r"\b"
        for match in re.finditer(pattern, text_lower):
            span = match.span()
            if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                matched_spans.add(span)
                original_str = text[span[0]:span[1]]
                entities.append({
                    "entity": original_str,
                    "type": "RESTAURANT",
                    "start": span[0],
                    "end": span[1],
                    "color": ENTITY_COLORS["RESTAURANT"]
                })

    # Regex heuristic for restaurant names like "X Restaurant", "Y Cafe", "Z Bistro"
    rest_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9\']*(?:\s+[A-Z][a-zA-Z0-9\']*)*\s+(?:Restaurant|Cafe|Bistro|Pizzeria|Diner|Dhaba|Kitchen|Lounge|Bakery|Grill))\b')
    for match in rest_pattern.finditer(text):
        span = match.span()
        if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
            matched_spans.add(span)
            entities.append({
                "entity": match.group(0),
                "type": "RESTAURANT",
                "start": span[0],
                "end": span[1],
                "color": ENTITY_COLORS["RESTAURANT"]
            })

    # 2. Match Known Food items
    for f in sorted(KNOWN_FOOD_ITEMS, key=len, reverse=True):
        pattern = r"\b" + re.escape(f) + r"\b"
        for match in re.finditer(pattern, text_lower):
            span = match.span()
            if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                matched_spans.add(span)
                original_str = text[span[0]:span[1]]
                entities.append({
                    "entity": original_str,
                    "type": "FOOD",
                    "start": span[0],
                    "end": span[1],
                    "color": ENTITY_COLORS["FOOD"]
                })

    # 3. Match Known Locations
    for loc in sorted(KNOWN_LOCATIONS, key=len, reverse=True):
        pattern = r"\b" + re.escape(loc) + r"\b"
        for match in re.finditer(pattern, text_lower):
            span = match.span()
            if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                matched_spans.add(span)
                original_str = text[span[0]:span[1]]
                entities.append({
                    "entity": original_str,
                    "type": "LOCATION",
                    "start": span[0],
                    "end": span[1],
                    "color": ENTITY_COLORS["LOCATION"]
                })

    # 4. Match Known Organizations
    for org in sorted(KNOWN_ORGANIZATIONS, key=len, reverse=True):
        pattern = r"\b" + re.escape(org) + r"\b"
        for match in re.finditer(pattern, text_lower):
            span = match.span()
            if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                matched_spans.add(span)
                original_str = text[span[0]:span[1]]
                entities.append({
                    "entity": original_str,
                    "type": "ORGANIZATION",
                    "start": span[0],
                    "end": span[1],
                    "color": ENTITY_COLORS["ORGANIZATION"]
                })

    # 5. Match Dates and Days
    for d in sorted(KNOWN_DATES, key=len, reverse=True):
        pattern = r"\b" + re.escape(d) + r"\b"
        for match in re.finditer(pattern, text_lower):
            span = match.span()
            if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                matched_spans.add(span)
                original_str = text[span[0]:span[1]]
                entities.append({
                    "entity": original_str,
                    "type": "DATE",
                    "start": span[0],
                    "end": span[1],
                    "color": ENTITY_COLORS["DATE"]
                })

    # 6. Supplement with NLTK ne_chunk for Person & General Entities
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    chunked = ne_chunk(tagged)

    for subtree in chunked:
        if isinstance(subtree, Tree):
            ne_label = subtree.label()
            ne_text = " ".join([leaf[0] for leaf in subtree.leaves()])
            pattern = r"\b" + re.escape(ne_text) + r"\b"
            for match in re.finditer(pattern, text):
                span = match.span()
                if not any(s <= span[0] and span[1] <= e for s, e in matched_spans):
                    mapped_type = "PERSON" if ne_label in ["PERSON"] else ("LOCATION" if ne_label in ["GPE", "LOCATION"] else "ORGANIZATION")
                    matched_spans.add(span)
                    entities.append({
                        "entity": ne_text,
                        "type": mapped_type,
                        "start": span[0],
                        "end": span[1],
                        "color": ENTITY_COLORS.get(mapped_type, "#757575")
                    })

    # Sort entities by appearance in text
    entities = sorted(entities, key=lambda x: x["start"])

    # Generate HTML annotated visualization string
    annotated_html = ""
    last_idx = 0
    for ent in entities:
        annotated_html += text[last_idx:ent["start"]]
        annotated_html += (
            f'<span style="background-color: {ent["color"]}; color: white; padding: 2px 6px; '
            f'border-radius: 4px; font-weight: 500; font-size: 0.9em; margin: 0 2px;">'
            f'{ent["entity"]} <sub style="font-size: 0.7em; opacity: 0.85;">{ent["type"]}</sub></span>'
        )
        last_idx = ent["end"]
    annotated_html += text[last_idx:]

    type_counts = {}
    for ent in entities:
        type_counts[ent["type"]] = type_counts.get(ent["type"], 0) + 1

    return {
        "entities": entities,
        "total_entities": len(entities),
        "entity_counts": type_counts,
        "annotated_html": annotated_html
    }


def run_experiment_8(text: str) -> Dict[str, Any]:
    """Runs Experiment 8 on a given input string."""
    return extract_entities(text)
