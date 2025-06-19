"""
Common ASL signs organized by categories.
"""

# Common ASL signs with their categories
SIGN_VOCABULARY = {
    "Greetings": [
        "Hello",
        "Goodbye",
        "Thank you",
        "Please",
        "Nice to meet you",
        "How are you",
    ],
    "Numbers": [
        "One",
        "Two",
        "Three",
        "Four",
        "Five",
        "Six",
        "Seven",
        "Eight",
        "Nine",
        "Ten",
    ],
    "Colors": [
        "Red",
        "Blue",
        "Green",
        "Yellow",
        "Black",
        "White",
        "Purple",
        "Orange",
        "Brown",
        "Pink",
    ],
    "Family": [
        "Mother",
        "Father",
        "Sister",
        "Brother",
        "Baby",
        "Family",
        "Grandmother",
        "Grandfather",
    ],
    "Time": [
        "Morning",
        "Afternoon",
        "Evening",
        "Night",
        "Today",
        "Tomorrow",
        "Yesterday",
    ],
    "Common Phrases": [
        "Yes",
        "No",
        "Help",
        "Sorry",
        "Excuse me",
        "I love you",
        "Good",
        "Bad",
    ],
    "Food & Drink": [
        "Water",
        "Food",
        "Eat",
        "Drink",
        "Hungry",
        "Thirsty",
        "Breakfast",
        "Lunch",
        "Dinner",
    ],
    "Weather": [
        "Hot",
        "Cold",
        "Rain",
        "Snow",
        "Sun",
        "Wind",
        "Storm",
    ],
    "Emotions": [
        "Happy",
        "Sad",
        "Angry",
        "Tired",
        "Excited",
        "Scared",
        "Surprised",
    ],
}

def get_all_signs():
    """Returns a flat list of all signs."""
    return [sign for category in SIGN_VOCABULARY.values() for sign in category]

def get_sign_categories():
    """Returns a list of all categories."""
    return list(SIGN_VOCABULARY.keys())

def get_signs_in_category(category):
    """Returns all signs in a given category."""
    return SIGN_VOCABULARY.get(category, []) 