import re

# Routing Thresholds
SCORE_THRESHOLD = 0.38  # If abs(score) < 0.38, it's ambiguous
ENTROPY_THRESHOLD = 0.7  # If entropy > 0.7, model is uncertain

# Sarcasm Detection Patterns (Kenyan Context)
# Regex patterns to catch common sarcastic markers
SARCASM_PATTERNS = [
    r"(?i)\baki\s+wewe\b",  # "Aki wewe" (Oh you...)
    r"(?i)\bwueh\b",  # "Wueh" (Exclamation of shock/disbelief)
    r"(?i)\beeh\b",  # "Eeh" (Can be sarcastic agreement)
    r"(?i)\bkwani\s+ni\s+kesho\b",  # "Kwani ni kesho" (Is it tomorrow? - implying do it now/consequences)
    r"(?i)\bni\s+god\s+manze\b",  # "Ni God manze" (It's God man - can be sarcastic resignation)
    r"(?i)\bbora\s+uhai\b",  # "Bora uhai" (As long as there is life - resignation)
    r"(?i)\bmapema\s+ndio\s+best\b",  # "Mapema ndio best" (Early is best - often used ironically)
    r"(?i)\bkuna\s+mtu\s+ataumia\b",  # "Kuna mtu ataumia" (Someone will get hurt - warning/sarcasm)
    r"(?i)\bsijui\s+kama\s+unaelewa\b",  # "Sijui kama unaelewa" (I don't know if you understand - condescending)
    r"(?i)\b(haha){3,}\b",  # Long laughter often implies mockery
    r"(?i)\b(lo+l)\b",  # LOL
    r"(?i)\bclown\b",  # Clown emoji/text
    r"(?i)\b\[CLOWN\]\b",  # Clown emoji
]

COMPILED_SARCASM_PATTERNS = [re.compile(p) for p in SARCASM_PATTERNS]
