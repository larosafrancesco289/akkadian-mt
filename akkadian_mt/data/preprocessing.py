"""Text preprocessing for Akkadian transliterations and English translations.

Based on Kaggle Deep Past Challenge formatting guidelines:
https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/overview/dataset-instructions
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

# =============================================================================
# Character Mappings
# =============================================================================

# Subscript digit mapping (Unicode subscripts U+2080-U+2089 to regular digits)
SUBSCRIPT_MAP = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "ₓ": "x",  # Subscript x (U+2093)
}

# Training data uses Ḫ/ḫ (U+1E2A/U+1E2B) but test data uses H/h
H_MAP = {"Ḫ": "H", "ḫ": "h"}

# Alternative transliteration conventions (some editions use simplified shin/emphatic-t)
# Verified: these patterns have 0 occurrences in standard training data (120K+ rows)
# except 'au' which has 3 occurrences — negligible false-positive risk.
_SHIN_CONVENTION_MAP = [
    ("mup-", "ṭup-"),  # emphatic-t: mup → ṭup (0 standard occurrences)
    ("mup ", "ṭup "),
    ("aí", "ší"),  # shin + i (0 standard occurrences)
    ("aé", "šé"),  # shin + e (0 standard occurrences)
    ("aù", "šù"),  # shin + u-grave (0 standard occurrences)
    ("aa", "ša"),  # shin + a (0 standard occurrences as syllable)
    ("au", "šu"),  # shin + u (3 standard occurrences in 120K+ rows)
]

# Homophone subscript digits: scholarly notation like bi4, en6, il5, PUZUR4.
# These distinguish cuneiform signs with the same reading but don't affect
# translation. Test data lacks them (uses „ which gets stripped), so stripping
# from training aligns the two distributions.
_HOMOPHONE_SUBSCRIPT_PATTERN = re.compile(r"(?<=[a-zA-ZàáèéìíòóùúšṭṣŠḪ])([0-9])(?=[-\s]|$)")

# =============================================================================
# Regex Patterns
# =============================================================================

# Sumerograms: all-uppercase tokens, possibly with dots/digits
SUMEROGRAM_PATTERN = re.compile(r"^[A-ZÀ-Ž0-9.₀-₉]+$")

# Modern scribal notations to remove:
#   ! = certain reading
#   ? = questionable reading
#   / = line divider
#   : = Old Assyrian word divider
#   „ = low-9 quotation mark (U+201E), scribal notation in some texts
#   + = reading notation (e.g., me-+e-er)
SCRIBAL_NOTATION_PATTERN = re.compile(r"[!?/:„+]")

# Bracket-like markers for partially broken signs:
#   ˹ ˺ = half brackets (U+F121, U+F122)
#   ‹ › = single guillemets (U+2039, U+203A), sometimes used as half brackets
#   ⌈ ⌉ = ceiling brackets (U+2308, U+2309), alternative notation
HALF_BRACKET_PATTERN = re.compile(r"[˹˺‹›⌈⌉]")

# Scribal insertions: keep text inside angle brackets, remove the brackets
SCRIBAL_INSERTION_PATTERN = re.compile(r"<(?!big_gap>|gap>)([^>]*)>")

# Erroneous signs in double angle brackets: remove entirely
ERRONEOUS_SIGNS_PATTERN = re.compile(r"<<[^>]*>>")

# Gap patterns for damaged tablet sections:
#   [...], [... ...], standalone ..., or Unicode ellipsis … (U+2026)
BIG_GAP_PATTERN = re.compile(r"\[\.\.\.[^\]]*\]|\.\.\.|\u2026")

# Single broken sign: [x] or [X]
SINGLE_GAP_PATTERN = re.compile(r"\[x\]|\[X\]")

# Broken signs in square brackets: keep text, remove brackets
# Excludes [x] which is handled separately as a gap marker
SQUARE_BRACKET_TEXT_PATTERN = re.compile(r"\[([^\]x][^\]]*)\]")

# Whitespace normalization
WHITESPACE_PATTERN = re.compile(r"\s+")

# Break comments in parentheses: (large break), (N broken lines), etc. → <big_gap>
BREAK_COMMENT_PATTERN = re.compile(
    r"\(\s*(?:large\s+break|(?:\d+\s+)?broken\s+(?:lines?|area)|break)\s*\)",
    re.IGNORECASE,
)

# Standalone x (single broken sign outside brackets) → <gap>
# Uses word boundary to avoid matching x inside tokens like NINDAx
STANDALONE_X_PATTERN = re.compile(r"\bx\b")

# Curly-brace determinatives: {d}, {ki}, {m}, etc. → (d), (ki), (m)
CURLY_DETERMINATIVE_PATTERN = re.compile(r"\{([^}]+)\}")

# Matches metadata-like translations that should be filtered out
META_RE = re.compile(
    r"^[\d\s./:;,\-]+$|^\(.*\)$|^col\.\s|^obv\b|^rev\b|^seal\s+impression"
    r"|^\[+\s*\]+$|^broken$|^uninscribed$|^erased$|^rest broken|^traces|^too broken",
    re.IGNORECASE,
)

# Subscript normalization as a str.maketrans table (faster than dict-based _apply_char_map)
SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")

# Bare gap tokens (without angle brackets) — for backward compat with stripped data
_BARE_BIG_GAP_PATTERN = re.compile(r"\bbig_gap\b")
_BARE_GAP_PATTERN = re.compile(r"\bgap\b")


def _apply_char_map(text: str, char_map: dict[str, str]) -> str:
    """Apply a character mapping to replace characters in text."""
    for old, new in char_map.items():
        text = text.replace(old, new)
    return text


def normalize_whitespace(text: str) -> str:
    """Strip and collapse multiple whitespace to single spaces."""
    return WHITESPACE_PATTERN.sub(" ", text.strip())


def normalize_subscripts(text: str) -> str:
    """Convert Unicode subscript digits to regular digits."""
    return text.translate(SUBSCRIPT_TRANS)


def normalize_h(text: str) -> str:
    """Normalize Ḫ/ḫ to H/h for test data compatibility."""
    return _apply_char_map(text, H_MAP)


def normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalization."""
    return unicodedata.normalize("NFC", text)


def is_sumerogram(token: str) -> bool:
    """Check if a token is a Sumerogram (uppercase with possible dots/digits)."""
    return bool(SUMEROGRAM_PATTERN.match(token))


def strip_homophone_subscripts(text: str) -> str:
    """Strip homophone subscript digits from transliteration.

    Removes single trailing digits from syllable tokens (e.g., bi4 → bi,
    en6 → en, il5 → il). These are scholarly markers distinguishing cuneiform
    signs with identical readings — irrelevant for translation.
    """
    return _HOMOPHONE_SUBSCRIPT_PATTERN.sub("", text)


def normalize_shin_convention(text: str) -> str:
    """Normalize alternative shin/emphatic-t transliteration conventions.

    Some text editions use simplified conventions where š (shin) is written as
    plain 'a' in syllable contexts and ṭ (emphatic t) as 'm'. These patterns
    are verified to have zero or near-zero occurrences in standard training data,
    so mapping them to standard convention is safe and improves test-time alignment.
    """
    for old, new in _SHIN_CONVENTION_MAP:
        text = text.replace(old, new)
    return text


def remove_scribal_notations(text: str) -> str:
    """Remove modern scribal notations (!, ?, /, :, „, +) from text."""
    return SCRIBAL_NOTATION_PATTERN.sub("", text)


def normalize_determinative_format(text: str) -> str:
    """Normalize curly-brace determinatives {d} to parenthesized (d) format."""
    return CURLY_DETERMINATIVE_PATTERN.sub(r"(\1)", text)


def normalize_break_comments(text: str) -> str:
    """Replace parenthesized break comments with <big_gap>."""
    return BREAK_COMMENT_PATTERN.sub("<big_gap>", text)


def normalize_standalone_x(text: str) -> str:
    """Replace standalone x tokens (broken signs) with <gap>."""
    return STANDALONE_X_PATTERN.sub("<gap>", text)


def normalize_gaps(text: str) -> str:
    """Replace gap markers with standardized tokens (<gap> and <big_gap>)."""
    text = BIG_GAP_PATTERN.sub("<big_gap>", text)
    text = SINGLE_GAP_PATTERN.sub("<gap>", text)
    return text


def normalize_brackets(text: str) -> str:
    """Normalize bracket notations: remove markers, keep enclosed text where appropriate."""
    text = ERRONEOUS_SIGNS_PATTERN.sub("", text)  # <<text>> -> removed entirely
    text = SCRIBAL_INSERTION_PATTERN.sub(r"\1", text)  # <text> -> text
    text = SQUARE_BRACKET_TEXT_PATTERN.sub(r"\1", text)  # [text] -> text (not [x])
    text = HALF_BRACKET_PATTERN.sub("", text)  # ˹˺‹›⌈⌉ -> removed
    return text


# Determinative pattern: {d}, {m}, (d), (m), {ki}, (ki), etc.
# Matches both curly-brace and parenthesized formats with balanced brackets.
# Safe because break comments (containing spaces) are already converted to <big_gap>.
DETERMINATIVE_PATTERN = re.compile(r"\{[a-zA-ZÀ-Ž₀-₉0-9]+\}|\([a-zA-ZÀ-Ž₀-₉0-9]+\)")

# Sumerogram detection: all-uppercase tokens (possibly with dots, digits, hyphens)
SUMEROGRAM_TOKEN_RE = re.compile(r"^[A-ZÀ-Ž][A-ZÀ-Ž0-9.\-₀-₉]+$")


def strip_determinatives(text: str) -> str:
    """Remove Akkadian determinatives like {d}, {m}, {ki}, {f} from text."""
    return DETERMINATIVE_PATTERN.sub("", text)


def tag_sumerograms(text: str) -> str:
    """Wrap Sumerogram tokens with <SUM>...</SUM> markers."""
    tokens = text.split()
    result = []
    for token in tokens:
        # Skip gap markers
        if token in ("<gap>", "<big_gap>"):
            result.append(token)
        elif SUMEROGRAM_TOKEN_RE.match(token):
            result.append(f"<SUM>{token}</SUM>")
        else:
            result.append(token)
    return " ".join(result)


def remove_gap_markers(text: str) -> str:
    """Remove <gap> and <big_gap> tokens from text.

    Also removes bare ``big_gap``/``gap`` strings for backward compatibility
    with data that had its angle brackets stripped by earlier preprocessing.
    """
    text = text.replace("<big_gap>", "")
    text = text.replace("<gap>", "")
    text = _BARE_BIG_GAP_PATTERN.sub("", text)
    text = _BARE_GAP_PATTERN.sub("", text)
    return text


def normalize_transliteration(
    text: str,
    full_normalize: bool = True,
    *,
    do_strip_determinatives: bool = False,
    do_tag_sumerograms: bool = False,
    do_remove_gaps: bool = False,
    do_strip_homophone_subscripts: bool = True,
) -> str:
    """Normalize an Akkadian transliteration string.

    Args:
        text: Raw transliteration string.
        full_normalize: If True, apply all competition-recommended normalizations
            (scribal notations, brackets, gaps). If False, only apply basic
            normalization (Unicode, Ḫ->H, subscripts, whitespace).
        do_strip_determinatives: RQ1 ablation - remove determinatives.
        do_tag_sumerograms: RQ2 ablation - wrap Sumerograms with <SUM> tags.
        do_remove_gaps: RQ4 ablation - remove gap markers.
        do_strip_homophone_subscripts: Strip scholarly subscript digits (bi4→bi).
            Default True for train/test alignment. Set False for backward compat
            with models trained on subscripted data.
    """
    text = normalize_unicode(text)
    text = normalize_h(text)
    text = normalize_subscripts(text)

    if full_normalize:
        text = remove_scribal_notations(text)
        # Normalize alternative shin/emphatic-t conventions (after scribal
        # notation removal strips „ so subscript markers are gone)
        text = normalize_shin_convention(text)
        # Strip homophone subscript digits (bi4→bi, en6→en) to align with
        # test data which lacks these markers
        if do_strip_homophone_subscripts:
            text = strip_homophone_subscripts(text)
        # Normalize determinative format: {d} → (d) for consistency
        text = normalize_determinative_format(text)
        # Break comments → <big_gap> (before bracket removal eats parentheses)
        text = normalize_break_comments(text)
        # Erroneous signs: <<text>> → removed entirely
        text = ERRONEOUS_SIGNS_PATTERN.sub("", text)
        # Scribal insertions: <text> → text (protected from <gap>/<big_gap>)
        text = SCRIBAL_INSERTION_PATTERN.sub(r"\1", text)
        # Gap markers: [...] → <big_gap>, [x] → <gap> (BEFORE bracket strip)
        text = normalize_gaps(text)
        # Standalone x → <gap> (after gap markers, catches remaining bare x)
        text = normalize_standalone_x(text)
        # Square bracket text: [text] → text (AFTER gap conversion)
        text = SQUARE_BRACKET_TEXT_PATTERN.sub(r"\1", text)
        # Half brackets: ˹˺‹›⌈⌉ → removed
        text = HALF_BRACKET_PATTERN.sub("", text)

    # Ablation transforms (applied after base normalization)
    if do_strip_determinatives:
        text = strip_determinatives(text)
    if do_remove_gaps:
        text = remove_gap_markers(text)
    if do_tag_sumerograms:
        text = tag_sumerograms(text)

    return normalize_whitespace(text)


def normalize_minimal(text: str) -> str:
    """Apply minimal normalization suitable for corpus assembly.

    Applies only: Unicode NFC, Ḫ→H, subscript digits, whitespace collapse.
    Equivalent to ``normalize_transliteration(text, full_normalize=False)``.
    """
    return normalize_transliteration(text, full_normalize=False)


def clean_translation(text: str) -> str:
    """Clean an English translation string (Unicode normalization + whitespace cleanup)."""
    text = normalize_unicode(text)
    return normalize_whitespace(text)


def normalize_inference_transliterations(
    values: list[str],
    *,
    data_dir: str | Path = "data/raw",
    remove_gaps: bool = False,
    strip_determinatives: bool = False,
    tag_sumerograms: bool = False,
    strip_homophone_subscripts: bool = True,
    use_dictionary_gloss: bool = False,
    max_glosses: int = 8,
) -> list[str]:
    """Normalize inference inputs to match training-time source preprocessing."""
    normalized = [
        normalize_transliteration(
            value,
            do_remove_gaps=remove_gaps,
            do_strip_determinatives=strip_determinatives,
            do_tag_sumerograms=tag_sumerograms,
            do_strip_homophone_subscripts=strip_homophone_subscripts,
        )
        for value in values
    ]

    if not use_dictionary_gloss:
        return normalized

    # Import lazily to avoid a circular import with akkadian_mt.data.dictionary.
    from akkadian_mt.data.dictionary import augment_with_glosses, build_gloss_lookup

    gloss_lookup = build_gloss_lookup(data_dir)
    return [
        augment_with_glosses(value, gloss_lookup, max_glosses=max_glosses) for value in normalized
    ]


# Pre-compiled patterns for postprocessing model output
_REPEATED_WORDS_PATTERN = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEATED_BIGRAM_PATTERN = re.compile(r"\b((?:\w+\s+)\w+)(?:\s+\1\b)+")
_REPEATED_TRIGRAM_PATTERN = re.compile(r"\b((?:\w+\s+){2}\w+)(?:\s+\1\b)+")
_REPEATED_4GRAM_PATTERN = re.compile(r"\b((?:\w+\s+){3}\w+)(?:\s+\1\b)+")
_PUNCT_SPACE_PATTERN = re.compile(r"\s+([.,;:!?])")
_REPEATED_PUNCT_PATTERN = re.compile(r"([.,;:])\1+")

# Output gap normalization (model may produce gap-like text)
_OUTPUT_GAP_PATTERN = re.compile(r"\[x\]|\(x\)", re.IGNORECASE)
_OUTPUT_BIG_GAP_PATTERN = re.compile(r"\.{3,}|\u2026|\[\.\.\.[^\]]*\]")
_ADJACENT_GAP_PATTERN = re.compile(r"<gap>\s*<gap>")
_ADJACENT_BIG_GAP_PATTERN = re.compile(r"<big_gap>\s*<big_gap>")


def _remove_phrase_repeats(text: str, max_phrase_len: int = 12) -> str:
    """Remove repeated phrases of any length up to max_phrase_len words."""
    words = text.split()
    if len(words) < 4:
        return text
    for plen in range(min(max_phrase_len, len(words) // 2), 1, -1):
        i = 0
        result: list[str] = []
        while i < len(words):
            phrase = words[i : i + plen]
            if len(phrase) < plen:
                result.extend(words[i:])
                break
            j = i + plen
            while j + plen <= len(words) and words[j : j + plen] == phrase:
                j += plen
            result.extend(phrase)
            i = j if j > i + plen else i + plen
        words = result
    return " ".join(words)


def postprocess_translation(text: str) -> str:
    """Post-process a model-generated English translation.

    Applies fixes to remove common MT artifacts:
    - Normalize gap-like patterns in output
    - Repeated phrases (up to 12 words) and n-grams (up to 4-grams)
    - Spurious spaces before punctuation
    - Repeated punctuation marks
    - Trailing fragment trimming

    NOTE: Does NOT strip chars like !?()[]"<>/ — references contain these.
    Does NOT convert fractions — references use decimals (0.5), not 1/2.
    Does NOT remove annotations like (fem. plur.) — references contain these.
    """
    if not text or not text.strip():
        return text

    # --- Strip gap markers (references never contain them) ---
    text = _OUTPUT_BIG_GAP_PATTERN.sub("", text)
    text = _OUTPUT_GAP_PATTERN.sub("", text)
    text = text.replace("<big_gap>", "")
    text = text.replace("<gap>", "")

    # --- Remove repeated phrases (up to 12 words) ---
    text = _remove_phrase_repeats(text)

    # --- Remove repeated n-grams (4-grams down to unigrams) ---
    text = _REPEATED_4GRAM_PATTERN.sub(r"\1", text)
    text = _REPEATED_TRIGRAM_PATTERN.sub(r"\1", text)
    text = _REPEATED_BIGRAM_PATTERN.sub(r"\1", text)
    text = _REPEATED_WORDS_PATTERN.sub(r"\1", text)

    # --- Fix spurious spaces before punctuation: "word ." -> "word." ---
    text = _PUNCT_SPACE_PATTERN.sub(r"\1", text)

    # --- Remove repeated punctuation: ".." -> "." ---
    text = _REPEATED_PUNCT_PATTERN.sub(r"\1", text)

    # --- Trim trailing fragments (>100 chars ending mid-word) ---
    text = text.strip()
    if len(text) > 100 and text[-1].isalpha():
        # Find the last sentence-ending punctuation and trim after it
        last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last_punct > 0:
            text = text[: last_punct + 1]

    # --- Strip leading/trailing hyphens ---
    text = text.strip("-").strip()

    return normalize_whitespace(text)
