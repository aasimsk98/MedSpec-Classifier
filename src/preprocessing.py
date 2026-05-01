"""
preprocessing.py
----------------
Shared text preprocessing utilities used across all three models.
Extracted from BioM_BERT.ipynb and demo.ipynb.
"""


def smart_truncate_words(text: str, total_limit: int = 350,
                          front: int = 175, back: int = 175) -> str:
    """
    Truncates long transcriptions by keeping the first `front` words
    and the last `back` words rather than blindly cutting at 512 tokens.

    Args:
        text        : raw transcription string
        total_limit : if word count <= this, return as-is (no truncation)
        front       : number of words to keep from the beginning
        back        : number of words to keep from the end

    Returns:
        Truncated string (or original if short enough)
    """
    words = text.split()
    if len(words) <= total_limit:
        return ' '.join(words)
    return ' '.join(words[:front] + words[-back:])
