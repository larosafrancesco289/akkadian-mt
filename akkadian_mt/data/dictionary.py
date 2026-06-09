"""Loaders for auxiliary data: eBL Dictionary, OA Lexicon, Sentences_Oare."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from akkadian_mt.data.preprocessing import SUBSCRIPT_TRANS

logger = logging.getLogger(__name__)


def load_ebl_dictionary(data_dir: str | Path) -> pd.DataFrame:
    """Load the eBL Dictionary (word → definition mappings).

    Returns DataFrame with columns: word, definition, derived_from
    """
    path = Path(data_dir) / "eBL_Dictionary.csv"
    df = pd.read_csv(path)
    logger.info("Loaded eBL Dictionary: %d entries", len(df))
    return df


def load_oa_lexicon(data_dir: str | Path) -> pd.DataFrame:
    """Load the Old Assyrian Lexicon with eBL links.

    Returns DataFrame with columns: type, form, norm, lexeme, eBL, ...
    """
    path = Path(data_dir) / "OA_Lexicon_eBL.csv"
    df = pd.read_csv(path)
    logger.info("Loaded OA Lexicon: %d entries", len(df))
    return df


def load_sentences_oare(data_dir: str | Path) -> pd.DataFrame:
    """Load Sentences_Oare (extra sentence-translation pairs).

    This file contains ~9.7k additional training examples beyond the main
    train.csv (which has only ~1.5k). Critical for data augmentation.

    Returns DataFrame with at minimum: translation column.
    Note: this file does NOT have a dedicated transliteration column —
    the transliteration must be reconstructed or the file used for
    translation-side augmentation only.
    """
    path = Path(data_dir) / "Sentences_Oare_FirstWord_LinNum.csv"
    df = pd.read_csv(path)
    logger.info("Loaded Sentences_Oare: %d entries", len(df))
    return df


_SUMEROGRAM_RE = re.compile(r"^[A-ZÀ-Ž0-9.]+$")
_ROMAN_SUFFIX_RE = re.compile(r"\s+[IVX]+$")
_SKIP_TOKENS = frozenset(("<gap>", "<big_gap>"))


def _normalize_form(form: str) -> str:
    """Normalize a transliteration form for dictionary matching."""
    form = form.strip().lower()
    form = form.translate(SUBSCRIPT_TRANS)
    form = form.replace("\u1e2b", "h")  # ḫ -> h
    return form


def _extract_short_gloss(definition: str) -> str:
    """Extract a short English gloss from a scholarly eBL definition.

    Looks for the first quoted string and truncates at the first semicolon or comma.
    """
    if not definition or not isinstance(definition, str):
        return ""
    match = re.search(r'"([^"]+)"', definition)
    if not match:
        return ""
    gloss = match.group(1)
    for sep in (";", ","):
        if sep in gloss:
            gloss = gloss[: gloss.index(sep)]
    return gloss.strip()


def build_gloss_lookup(data_dir: str | Path) -> dict[str, str]:
    """Build a token → short English gloss lookup from OA Lexicon + eBL Dictionary.

    Strategy:
    1. From OA Lexicon: form → lexeme mapping (handles inflected forms)
    2. From eBL Dictionary: lexeme → definition → short gloss
    3. Combined: transliteration token → short English gloss

    Returns:
        Dict mapping normalized transliteration forms to short English glosses.
    """
    data_dir = Path(data_dir)

    # Build eBL lookup: word -> short gloss (stored under multiple key variants)
    ebl = pd.read_csv(data_dir / "eBL_Dictionary.csv")
    ebl_lookup: dict[str, str] = {}
    for row in ebl.itertuples(index=False):
        word = str(row.word).strip()
        definition = row.definition
        if pd.isna(definition):
            continue
        gloss = _extract_short_gloss(str(definition))
        if not gloss:
            continue
        ebl_lookup[word.lower()] = gloss
        ebl_lookup[_normalize_form(word)] = gloss
        base = _ROMAN_SUFFIX_RE.sub("", word).strip()
        if base != word:
            ebl_lookup.setdefault(base.lower(), gloss)
            ebl_lookup.setdefault(_normalize_form(base), gloss)

    logger.info("eBL glosses extracted: %d entries with short glosses", len(ebl_lookup))

    # Build form -> gloss mapping via OA Lexicon (form -> lexeme -> eBL gloss)
    oa = pd.read_csv(data_dir / "OA_Lexicon_eBL.csv")
    oa_words = oa[oa["type"] == "word"]
    form_to_gloss: dict[str, str] = {}

    for row in oa_words.itertuples(index=False):
        form = str(row.form).strip()
        lexeme = str(row.lexeme).strip()
        if not form or not lexeme or form == "nan" or lexeme == "nan":
            continue

        gloss = ebl_lookup.get(lexeme.lower()) or ebl_lookup.get(_normalize_form(lexeme))
        if not gloss:
            base_lex = _ROMAN_SUFFIX_RE.sub("", lexeme.lower())
            gloss = ebl_lookup.get(base_lex) or ebl_lookup.get(_normalize_form(base_lex))

        if gloss:
            form_to_gloss.setdefault(_normalize_form(form), gloss)

    logger.info(
        "Dictionary lookup built: %d form->gloss entries (from %d OA word forms)",
        len(form_to_gloss),
        len(oa_words),
    )

    return form_to_gloss


def augment_with_glosses(
    transliteration: str,
    gloss_lookup: dict[str, str],
    max_glosses: int = 8,
) -> str:
    """Append dictionary glosses to a transliteration as hints.

    Produces format: "original text || token1: gloss1; token2: gloss2"

    Args:
        transliteration: Normalized Akkadian transliteration.
        gloss_lookup: Dict from build_gloss_lookup().
        max_glosses: Maximum number of glosses to append.

    Returns:
        Augmented transliteration with gloss hints appended.
    """
    tokens = transliteration.split()
    glosses: list[str] = []

    for token in tokens:
        if len(glosses) >= max_glosses:
            break
        if token in _SKIP_TOKENS or _SUMEROGRAM_RE.match(token):
            continue
        gloss = gloss_lookup.get(_normalize_form(token))
        if gloss:
            glosses.append(f"{token}: {gloss}")

    if not glosses:
        return transliteration

    return transliteration + " || " + "; ".join(glosses)
