#!/usr/bin/env python
"""Extract translations from publications.csv and match them to published_texts.csv.

This script:
1. Extracts translations from OCR'd text in publications.csv
2. Matches extracted translations to transliterations in published_texts.csv
3. Handles multi-language translations (French, German, Turkish, etc.)
4. Creates sentence-level alignments from the extracted data

Usage:
    python scripts/extract_publication_translations.py [--output OUTPUT_CSV] [--sample N]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from langdetect import DetectorFactory, LangDetectException
from langdetect import detect as _langdetect_detect

sys.path.insert(0, str(Path(__file__).parent.parent))

from akkadian_mt.data.preprocessing import clean_translation, normalize_transliteration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Keep publication-language filtering stable across rebuilds.
DetectorFactory.seed = 0


# =============================================================================
# Translation Extraction Patterns
# =============================================================================

# Pattern for quoted translations: "English text" or „German text"
QUOTED_TRANSLATION_PATTERN = re.compile(
    r'["„]([^"„"]{10,200})[""]',  # 10-200 chars between quotes
    re.IGNORECASE,
)

# Pattern for line-by-line translations: "line 5: translation" or "l. 5 translation"
LINE_TRANSLATION_PATTERN = re.compile(
    r'(?:line|lines|l\.|l\s+)\s*[\d\']+(?:\s*[-–]\s*[\d\']+)?\s*[:–]\s*([A-Z][^.!?]{10,200})',
    re.IGNORECASE,
)

# Pattern for translation markers: "translation:", "translated as:", etc.
TRANSLATION_MARKER_PATTERN = re.compile(
    r'translat(?:ion|ed|ing)?\s*[:–]\s*([A-Z][^.!?]{10,200})',
    re.IGNORECASE,
)

# Pattern for explicit "Translation." blocks (common in editions)
TRANSLATION_BLOCK_PATTERN = re.compile(
    r'\bTranslation\b\s*[\.:–]\s*([A-Z][\s\S]{10,300}?)(?=\n\s*\n|$)',
    re.IGNORECASE,
)

# Pattern for parenthetical translations: (English text) or [English text]
PARENTHETICAL_TRANSLATION_PATTERN = re.compile(
    r'[\(\[]([A-Z][a-z][^\)\]]{10,150})[\)\]]',
    re.IGNORECASE,
)


# =============================================================================
# Identifier Matching
# =============================================================================

# Pre-compiled identifier patterns (called ~31K times in the full pipeline)
_CDLI_ID_RE = re.compile(r'\bP\d{6}\b')
_EBL_ID_RE = re.compile(r'\beBL[-\s]?[\w-]+', re.IGNORECASE)
_KT_EXCAVATION_RE = re.compile(r'\bKt\s+\w{1,3}/k\s+\d+[a-z]?\b', re.IGNORECASE)

_CATALOG_PREFIXES = (
    r'AKT|CCT|TC|ICK|BIN|KTS|VS|TMH|KTK|POAT|ATHE|KBo|OIP|KUG|KTH|'
    r'CTMMA|KTP|CUSAS|TTC|KTB|HAM|HSS|TPAK|OAA|SUP|NMS|'
    r'RA|Or|OrNS|AnOr|JCS|AAA|EL|ArAn|UF|Anatolica'
)
_CATALOG_ARABIC_RE = re.compile(
    rf'\b(?:{_CATALOG_PREFIXES})\s+\d+[,\s]+(?:n\.\s*)?\d+[a-z]?\b', re.IGNORECASE,
)
_CATALOG_ROMAN_RE = re.compile(
    rf'\b(?:{_CATALOG_PREFIXES})\s+(?:I{{1,3}}|IV|VI?I{{0,2}}|IX|X)\s+\d+[a-z]?\b', re.IGNORECASE,
)
_STANDALONE_CATALOG_RE = re.compile(
    r'\b(?:Prag|Chantre|Sadberk|Nesr|Adana|Brussel|Landsberger)\s+(?:\w+\s+)?\d+[a-z]?\b', re.IGNORECASE,
)
_MUSEUM_RE = re.compile(
    r'\b(?:BM|NBC|MAH|AO|CBS|MMA|MRAH|WAG|Bod|Ka|Ist\s+Ka|MP|Kayseri)\s+\d+[a-z]?(?:\s*bis)?\b', re.IGNORECASE,
)
_SIMPLE_MUSEUM_RE = re.compile(r'\b(?:KTS|MP)\s+\d+[a-z]?\b', re.IGNORECASE)
_OATP_RE = re.compile(r'\bOATP[-\s]?[\w-]+\b', re.IGNORECASE)

# Roman numeral → Arabic conversion (used by _normalize_extracted_identifier_for_keys)
_ROMAN_TO_ARABIC = {
    'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
    'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
}

# Full catalog name → abbreviation (LLM sometimes outputs full names)
_FULL_NAME_TO_ABBREV: dict[str, str] = {
    'ankara kultepe tabletleri': 'AKT',
    'cuneiform cappadocian tablets': 'CCT',
    'tablettes cappadociennes': 'TC',
    'inscriptions cuneiformes du kultepe': 'ICK',
    'babylonian inscriptions in the collection of j. b. nies': 'BIN',
    'keilschrifttexte aus assur': 'KTS',
    'vorderasiatische schriftdenkmaler': 'VS',
}


def extract_identifiers_from_text(text: str) -> dict[str, list[str]]:
    """Extract potential identifiers from OCR text.

    Looks for:
    - CDLI IDs: P\\d{6} (e.g., P361099)
    - eBL IDs: eBL\\d+ or similar patterns
    - Excavation numbers: Kt 94/k 840, Kt n/k 602, etc.
    - Publication catalog numbers: ICK 1 146, CCT 6 17a, AKT 2 57, etc.
    - Museum numbers: BM 115099, NBC 1662, Ka 185, etc.

    Returns:
        Dictionary mapping identifier types to lists of found IDs
    """
    identifiers: dict[str, list[str]] = defaultdict(list)

    identifiers['cdli_id'].extend(_CDLI_ID_RE.findall(text))
    identifiers['eBL_id'].extend(_EBL_ID_RE.findall(text))
    identifiers['excavation_no'].extend(_KT_EXCAVATION_RE.findall(text))
    identifiers['publication_catalog'].extend(_CATALOG_ARABIC_RE.findall(text))
    identifiers['publication_catalog'].extend(_CATALOG_ROMAN_RE.findall(text))
    identifiers['publication_catalog'].extend(_STANDALONE_CATALOG_RE.findall(text))
    identifiers['inventory_position'].extend(_MUSEUM_RE.findall(text))
    identifiers['inventory_position'].extend(_SIMPLE_MUSEUM_RE.findall(text))
    identifiers['oatp_key'].extend(_OATP_RE.findall(text))

    return identifiers


def normalize_identifier(id_str: str | float) -> str | None:
    """Normalize an identifier string for matching."""
    if pd.isna(id_str):
        return None
    id_str = str(id_str).strip()
    if not id_str or id_str.lower() in ('nan', 'none', ''):
        return None
    # Normalize whitespace and case
    id_str = re.sub(r'\s+', ' ', id_str)
    return id_str


def normalize_identifier_key(id_str: str) -> str:
    """Normalize an identifier into a match key that is robust to OCR formatting.

    Examples:
      - "P 361099" -> "p361099"
      - "ICK 1 146" -> "ick1146"
      - "eBL-12345" -> "ebl12345"
    """
    s = (id_str or "").strip().lower()
    s = s.replace("–", "").replace("—", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def split_identifier_list(id_str: str | float) -> list[str]:
    """Split a pipe-separated identifier list and normalize."""
    if pd.isna(id_str):
        return []
    id_str = str(id_str).strip()
    if not id_str or id_str.lower() in ('nan', 'none', ''):
        return []
    # Split by pipe and normalize
    return [normalize_identifier(x) for x in id_str.split('|') if normalize_identifier(x)]


def _parse_label_identifiers(label: str) -> list[str]:
    """Parse identifier(s) from a label like 'Cuneiform Tablet CCT 6 17a (BM 115099)'.

    Returns a list of identifier strings (e.g. ['CCT 6 17a', 'BM 115099']).
    """
    if not label:
        return []
    # Strip "Cuneiform Tablet " prefix
    label = re.sub(r'^Cuneiform Tablet\s+', '', label, flags=re.IGNORECASE).strip()
    parts = []
    # Extract parenthetical (museum number)
    m = re.search(r'\(([^)]+)\)', label)
    if m:
        parts.append(m.group(1).strip())
        label = label[:m.start()].strip()
    if label:
        parts.append(label)
    return parts


def build_published_texts_id_index(published_texts_df: pd.DataFrame) -> dict[str, list[str]]:
    """Build an index from normalized identifier key -> list of oare_ids.

    Uses multiple columns: cdli_id, eBL_id, publication_catalog, inventory_position,
    aliases, oatp_key, label, excavation_no.
    """
    cols = [
        "cdli_id",
        "eBL_id",
        "publication_catalog",
        "inventory_position",
        "aliases",
        "oatp_key",
        "excavation_no",
    ]
    cols = [c for c in cols if c in published_texts_df.columns]
    has_label = "label" in published_texts_df.columns

    index: dict[str, list[str]] = defaultdict(list)
    for _, row in published_texts_df.iterrows():
        oare_id = row.get("oare_id")
        if pd.isna(oare_id):
            continue
        oare_id = str(oare_id)

        # Standard columns
        for col in cols:
            raw = row.get(col)
            if pd.isna(raw):
                continue
            parts = split_identifier_list(raw) if "|" in str(raw) else [normalize_identifier(raw)]
            for part in parts:
                if not part:
                    continue
                key = normalize_identifier_key(part)
                if not key:
                    continue
                index[key].append(oare_id)

        # Label column: parse "Cuneiform Tablet CCT 6 17a (BM 115099)" → index both
        if has_label:
            label_raw = row.get("label")
            if not pd.isna(label_raw):
                for label_id in _parse_label_identifiers(str(label_raw)):
                    key = normalize_identifier_key(label_id)
                    if key:
                        index[key].append(oare_id)

    # Add base-form keys for excavation numbers with A/B/C suffixes.
    # "Kt 94/k 1146A" → also index "kt94k1146" so LLM output without suffix matches.
    suffix_extras: dict[str, list[str]] = defaultdict(list)
    for key, oare_ids in index.items():
        m = re.match(r'^(kt\w+k\d+)[a-c]$', key)
        if m:
            base = m.group(1)
            suffix_extras[base].extend(oare_ids)
    for base, oare_ids in suffix_extras.items():
        index[base].extend(oare_ids)

    # Deduplicate lists
    for k, v in list(index.items()):
        index[k] = sorted(set(v))
    return index


def _normalize_extracted_identifier_for_keys(extracted: str) -> list[str]:
    """Generate one or more normalized keys for an extracted OCR identifier.

    Handles common LLM/OCR output patterns:
    - Line-range suffixes: "kt 94/k 840: 2-23" → "kt94k840"
    - Parenthetical refs: "Larsen (2002, 168: 3-7)" → skipped (author ref)
    - Comma in numbers: "AKT 2, 57" → "akt257"
    - Roman numerals: "AKT II 67" → "akt267" (try both forms)
    """
    extracted = extracted.strip()
    if not extracted:
        return []

    # Strip line-range suffixes: ": 2-23", ": 5-14", ": 1-5"
    cleaned = re.sub(r':\s*\d+(?:\s*[-–]\s*\d+)?\s*$', '', extracted).strip()
    # Strip trailing parenthetical page/line refs: "(2002,168: 3-7)"
    cleaned = re.sub(r'\([^)]*\)\s*$', '', cleaned).strip()

    keys = set()
    keys.add(normalize_identifier_key(cleaned))

    # Also try the raw form (e.g. if suffix stripping was wrong)
    keys.add(normalize_identifier_key(extracted))

    # Special case: CDLI IDs can appear as "P 361099" or "P-361099"
    m = re.search(r"\bP\s*[-]?\s*(\d{6})\b", extracted, re.IGNORECASE)
    if m:
        keys.add(normalize_identifier_key("P" + m.group(1)))

    # Roman numeral → Arabic conversion for catalog refs
    m_roman = re.match(r'^([A-Za-z]+)\s+(I{1,3}|IV|VI{0,3}|IX|X)\b(.*)$', cleaned)
    if m_roman:
        prefix, roman, rest = m_roman.groups()
        arabic = _ROMAN_TO_ARABIC.get(roman.upper(), roman)
        keys.add(normalize_identifier_key(f"{prefix} {arabic}{rest}"))

    # Full-name → abbreviation mapping (LLM sometimes gives full names)
    # Normalize Unicode (İ→i, ü→u, é→e, etc.) for robust matching
    cleaned_lower = unicodedata.normalize('NFKD', cleaned).lower()
    cleaned_lower = ''.join(c for c in cleaned_lower if not unicodedata.combining(c))
    for full, abbrev in _FULL_NAME_TO_ABBREV.items():
        if cleaned_lower.startswith(full):
            rest_part = cleaned_lower[len(full):].strip()
            # Extract roman numeral + number from remainder
            m_rest = re.match(r'(I{1,3}|IV|VI{0,3}|IX|X|\d+)\s+(\d+[a-z]?)', rest_part, re.IGNORECASE)
            if m_rest:
                vol, num = m_rest.groups()
                vol_arabic = _ROMAN_TO_ARABIC.get(vol.upper(), vol)
                keys.add(normalize_identifier_key(f"{abbrev} {vol_arabic} {num}"))

    return [k for k in keys if k]

def match_text_to_published_texts(
    extracted_ids: dict[str, list[str]],
    published_texts_df: pd.DataFrame,
    *,
    id_index: dict[str, list[str]] | None = None,
) -> list[str]:
    """Match extracted identifiers to published_texts.csv rows.

    Args:
        extracted_ids: Identifiers extracted from page text
        published_texts_df: Published texts dataframe (unused if id_index provided)
        id_index: Pre-built identifier index (avoids rebuilding per call)

    Returns:
        List of oare_ids that match the extracted identifiers
    """
    if id_index is None:
        id_index = build_published_texts_id_index(published_texts_df)

    matched: set[str] = set()
    for _, extracted_values in extracted_ids.items():
        for extracted_id in extracted_values:
            for key in _normalize_extracted_identifier_for_keys(str(extracted_id)):
                for oare_id in id_index.get(key, []):
                    matched.add(oare_id)

    return sorted(matched)


# =============================================================================
# Translation Extraction
# =============================================================================

def detect_language(text: str) -> str:
    """Detect the language of a translation text using langdetect.

    Returns:
        ISO 639-1 language code (e.g. 'en', 'fr', 'de', 'it', 'nl', 'tr')
        or 'unknown' if detection fails.
    """
    text = (text or "").strip()
    if len(text) < 10:
        return "unknown"
    try:
        return _langdetect_detect(text)
    except LangDetectException:
        return "unknown"


# =============================================================================
# Confidence Scoring (heuristics)
# =============================================================================

_EN_COMMON_WORDS = re.compile(
    r"\b(?:the|and|to|of|in|for|with|on|at|by|from|this|that|is|are|was|were|be|as|it|"
    r"he|she|they|we|you|i|his|her|their|our|your|not|have|has|had|will|shall|may|"
    r"said|say|send|give|take|pay|silver|mina|minas|shekel|shekels|witness(?:es)?|"
    r"son|seal|month|year)\b",
    re.IGNORECASE,
)

_CITATION_RE = re.compile(
    r"(?:^|(?<=\W))(?:cf\.|ibid\.|id\.|op\.?\s*c?it\.|fig\.|pl\.|no\.|n\.)"
    r"(?=\W|$)|"
    r"\b(?:dercksen|michel|gelb|lewy|gurney|barjamovic|larsen|von\s+soden)\b|"
    r"\b(?:KTK|ICK|RA|BIN|OIP|TCL|AOATT|Or|AHw|AbB|ARMT|ARM[T]?)\b|"
    r"(?:^|(?<=\W))p{1,2}\.\s*\d+[a-z]?(?=\W|$)|"
    r"\b(?:19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

_SCHOLARLY_PHRASE_RE = re.compile(
    r"\b(?:as shown in|see (?:also )?|following (?:a )?suggestion|"
    r"previous literature|with previous literature|proposes|proposal|"
    r"chapter|ch\.\s*\w+|section|sec\.\s*\w+|appendix|app\.\s*\w+|"
    r"note\s+\d+|n\.\s*\d+|pp?\.\s*\d+|lines?\s+\d+|col\.\s*\w+)\b",
    re.IGNORECASE,
)

_META_PHRASE_RE = re.compile(
    r"(?:^|(?<=\W))(?:obv\.|rev\.|col\.|lo\.\s*e\.)(?=\W|$)|"
    r"\b(?:seal\s+impression|seal\s+impressions|broken|uninscribed|erased|not\s+inscribed|fragmentary)\b",
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r"\b(?:is called|is known as|refers to|designates|means\b|whose task|"
    r"as a means of|commercial enterprise|term\b|designation\b)\b",
    re.IGNORECASE,
)

_PHILOLOGY_RE = re.compile(
    r"\b(?:cannot be the subject of|it is taken here as|"
    r"with the ending|also in\s+\d+|occurrence of the|plural\b|sing\.\b|"
    r"read in (?:Assyrian|Akkadian)|written by|in the letter\b|"
    r"line\s+\d+[-–]\d+|lines\s+\d+[-–]\d+|"
    r"paradigm\b|durative\b|st-?form\b|umlaut\b|"
    r"\bGAG\b|\bCAD\b|\bCDA\b|"
    r"\breads?\b|\bSemitic\b|\broot\b|\bverb\b|\bconnect\b|\bmeaning\b|"
    r"\bto be published\b|\bpublished in\b)\b",
    re.IGNORECASE,
)

_APPARATUS_RE = re.compile(
    r"(?:^|(?<=\W))[a-z]\)\s|(?:^|(?<=\W))\d+\)\s|"
    r"\b(?:form should|should anyhow be added|added to)\b",
    re.IGNORECASE,
)

_TRANSLIT_GLYPHS_RE = re.compile(r"[₀-₉ₓšṣṭḫĀĒĪŪáàéèíìúù]|<gap>|<big_gap>", re.IGNORECASE)
_TRANSLIT_TOKEN_RE = re.compile(r"\b[a-z]{1,3}(?:-[a-z]{1,3}){2,}\b", re.IGNORECASE)


def score_translation_confidence(text: str, language: str) -> dict[str, Any]:
    """Heuristically score whether an extracted string is a usable English translation."""
    s = (text or "").strip()
    flags: list[str] = []
    if not s:
        return {"confidence": 0.0, "flags": ["empty"], "common_word_hits": 0, "alpha_fraction": 0.0}

    letters = sum(ch.isalpha() for ch in s)
    spaces = s.count(" ")
    alpha_frac = letters / max(len(s), 1)
    if alpha_frac < 0.45:
        flags.append("low_alpha_fraction")

    common_hits = len(_EN_COMMON_WORDS.findall(s))
    if common_hits == 0:
        flags.append("no_common_english_words")

    if _CITATION_RE.search(s):
        flags.append("citation_like")

    if _SCHOLARLY_PHRASE_RE.search(s):
        flags.append("scholarly_like")

    if _META_PHRASE_RE.search(s):
        flags.append("meta_like")

    if _DEFINITION_RE.search(s):
        flags.append("definition_like")

    if _PHILOLOGY_RE.search(s):
        flags.append("philology_like")

    if _APPARATUS_RE.search(s):
        flags.append("apparatus_like")

    hyphen_rate = s.count("-") / max(spaces + 1, 1)
    if _TRANSLIT_GLYPHS_RE.search(s) or _TRANSLIT_TOKEN_RE.search(s) or hyphen_rate > 0.8:
        flags.append("transliteration_like")

    if "\n" in s:
        flags.append("contains_newlines")

    conf = 0.5
    if language == "en":
        conf += 0.15
    if common_hits >= 2:
        conf += 0.15
    if common_hits >= 5:
        conf += 0.10
    if alpha_frac >= 0.60:
        conf += 0.10
    if "citation_like" in flags:
        conf -= 0.25
    if "scholarly_like" in flags:
        conf -= 0.20
    if "meta_like" in flags:
        conf -= 0.35
    if "definition_like" in flags:
        conf -= 0.25
    if "philology_like" in flags:
        conf -= 0.30
    if "apparatus_like" in flags:
        conf -= 0.25
    if "transliteration_like" in flags:
        conf -= 0.35
    if "contains_newlines" in flags:
        conf -= 0.10
    if "low_alpha_fraction" in flags:
        conf -= 0.10
    if "no_common_english_words" in flags and language != "en":
        conf -= 0.10

    conf = max(0.0, min(1.0, conf))
    return {
        "confidence": conf,
        "flags": flags,
        "common_word_hits": common_hits,
        "alpha_fraction": alpha_frac,
    }


def extract_translations_from_text(text: str) -> list[dict[str, Any]]:
    """Extract potential translations from OCR text.

    Returns:
        List of dictionaries with keys: 'text', 'language', 'pattern_type', 'context'
    """
    translations = []

    _EXTRACTION_PATTERNS = [
        (QUOTED_TRANSLATION_PATTERN, 'quoted'),
        (LINE_TRANSLATION_PATTERN, 'line'),
        (TRANSLATION_MARKER_PATTERN, 'marker'),
        (TRANSLATION_BLOCK_PATTERN, 'translation_block'),
        (PARENTHETICAL_TRANSLATION_PATTERN, 'parenthetical'),
    ]

    for pattern, pattern_type in _EXTRACTION_PATTERNS:
        for match in pattern.finditer(text):
            trans_text = match.group(1).strip()
            # Translation blocks may span lines — collapse whitespace
            if pattern_type == 'translation_block':
                trans_text = re.sub(r"\s+", " ", trans_text)
            if len(trans_text) < 10:
                continue
            language = detect_language(trans_text)
            score = score_translation_confidence(trans_text, language)
            # Translation blocks use a wider context window
            if pattern_type == 'translation_block':
                ctx = text[max(0, match.start()-50):match.start()+min(400, len(text)-match.start())]
            else:
                ctx = text[max(0, match.start()-50):match.end()+50]
            translations.append({
                'text': trans_text,
                'language': language,
                'pattern_type': pattern_type,
                'context': ctx,
                'confidence': score["confidence"],
                'confidence_flags': "|".join(score["flags"]),
                'common_word_hits': score["common_word_hits"],
                'alpha_fraction': score["alpha_fraction"],
            })

    return translations


def clean_extracted_translation(text: str) -> str:
    """Clean an extracted translation text."""
    # Remove common OCR artifacts
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'^[^\w]+|[^\w]+$', '', text)  # Remove leading/trailing non-word chars
    text = text.strip()
    return clean_translation(text)


# =============================================================================
# Main Processing
# =============================================================================

def process_publications(
    data_dir: Path,
    output_path: Path,
    sample_size: int | None = None,
    min_translation_length: int = 10,
) -> None:
    """Main processing function to extract and match translations.

    Args:
        data_dir: Path to data directory containing CSV files
        output_path: Path to output CSV file
        sample_size: If provided, only process first N pages
        min_translation_length: Minimum length for extracted translations
    """
    logger.info("Loading data files...")

    # Load published_texts.csv
    pub_texts_path = data_dir / "published_texts.csv"
    if not pub_texts_path.exists():
        logger.error(f"published_texts.csv not found at {pub_texts_path}")
        return

    published_texts = pd.read_csv(pub_texts_path)
    logger.info(f"Loaded {len(published_texts)} published texts")

    logger.info("Building published_texts identifier index...")
    id_index = build_published_texts_id_index(published_texts)
    logger.info("Identifier index size: %d keys", len(id_index))

    # Build O(1) oare_id → row lookup (avoids O(n) boolean mask per lookup in main loop)
    _pub_texts_by_oare_id: dict[str, pd.Series] = {}
    for _, row in published_texts.iterrows():
        oid = row.get("oare_id")
        if not pd.isna(oid):
            _pub_texts_by_oare_id[str(oid)] = row

    # Load publications.csv
    pubs_path = data_dir / "publications.csv"
    if not pubs_path.exists():
        logger.error(f"publications.csv not found at {pubs_path}")
        return

    logger.info("Loading publications.csv (this may take a while for 554MB file)...")
    if sample_size:
        publications = pd.read_csv(pubs_path, nrows=sample_size)
        logger.info(f"Loaded sample of {len(publications)} publication pages")
    else:
        publications = pd.read_csv(pubs_path)
        logger.info(f"Loaded {len(publications)} publication pages")

    # Filter to pages with Akkadian
    akkadian_pages = publications[publications['has_akkadian']].copy()
    logger.info(f"Found {len(akkadian_pages)} pages with Akkadian")

    # Process pages
    all_extracted_pairs = []
    stats = {
        'pages_processed': 0,
        'pages_with_identifiers': 0,
        'pages_with_translations': 0,
        'translations_extracted': 0,
        'matched_to_published_texts': 0,
        'languages': defaultdict(int),
    }

    logger.info("Processing publication pages...")
    for idx, page_row in akkadian_pages.iterrows():
        stats['pages_processed'] += 1

        if stats['pages_processed'] % 1000 == 0:
            logger.info(f"Processed {stats['pages_processed']} pages... "
                       f"({stats['matched_to_published_texts']} matched pairs so far)")

        pdf_name = page_row.get('pdf_name', '')
        page_num = page_row.get('page', '')
        page_text = str(page_row.get('page_text', ''))

        if not page_text or len(page_text) < 50:
            continue

        # Extract identifiers from page text
        extracted_ids = extract_identifiers_from_text(page_text)
        if not any(extracted_ids.values()):
            continue

        stats['pages_with_identifiers'] += 1

        # Match to published_texts (use pre-built index)
        matched_oare_ids = match_text_to_published_texts(extracted_ids, published_texts, id_index=id_index)
        if not matched_oare_ids:
            continue

        # Extract translations from page text
        translations = extract_translations_from_text(page_text)
        if not translations:
            continue

        stats['pages_with_translations'] += 1

        # For each matched text and extracted translation, create a pair
        for oare_id in matched_oare_ids:
            pub_row = _pub_texts_by_oare_id.get(oare_id)
            if pub_row is None:
                continue

            transliteration = pub_row.get('transliteration', '')

            if pd.isna(transliteration) or not str(transliteration).strip():
                continue

            # Normalize transliteration
            transliteration_norm = normalize_transliteration(str(transliteration))

            # Create pairs with each extracted translation
            for trans_info in translations:
                trans_text = clean_extracted_translation(trans_info['text'])

                if len(trans_text) < min_translation_length:
                    continue

                stats['translations_extracted'] += 1
                stats['languages'][trans_info['language']] += 1
                # Re-score on the cleaned text (more stable than raw OCR snippet)
                score = score_translation_confidence(trans_text, trans_info['language'])

                all_extracted_pairs.append({
                    'oare_id': oare_id,
                    'transliteration': transliteration_norm,
                    'translation': trans_text,
                    'translation_language': trans_info['language'],
                    'extraction_pattern': trans_info['pattern_type'],
                    'pdf_name': pdf_name,
                    'page': page_num,
                    'source': 'publications_csv',
                    'translation_confidence': score["confidence"],
                    'translation_confidence_flags': "|".join(score["flags"]),
                    'translation_common_word_hits': score["common_word_hits"],
                    'translation_alpha_fraction': score["alpha_fraction"],
                    'matched_identifiers': '|'.join([
                        f"{k}:{v}" for k, values in extracted_ids.items()
                        for v in values
                    ]),
                    'extraction_context': trans_info.get('context', ''),
                })

                stats['matched_to_published_texts'] += 1

    # Create output DataFrame
    if not all_extracted_pairs:
        logger.warning("No translation pairs extracted!")
        return

    output_df = pd.DataFrame(all_extracted_pairs)

    # Remove duplicates (same oare_id + translation)
    initial_count = len(output_df)
    output_df = output_df.drop_duplicates(subset=['oare_id', 'translation'], keep='first')
    logger.info(f"Removed {initial_count - len(output_df)} duplicate pairs")

    # Save output
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(output_df)} extracted pairs to {output_path}")

    # Print statistics
    logger.info("\n=== Extraction Statistics ===")
    logger.info(f"Pages processed: {stats['pages_processed']}")
    logger.info(f"Pages with identifiers: {stats['pages_with_identifiers']}")
    logger.info(f"Pages with translations: {stats['pages_with_translations']}")
    logger.info(f"Translations extracted: {stats['translations_extracted']}")
    logger.info(f"Matched to published_texts: {stats['matched_to_published_texts']}")
    logger.info(f"Final unique pairs: {len(output_df)}")
    logger.info("\nLanguage distribution:")
    for lang, count in sorted(stats['languages'].items(), key=lambda x: -x[1]):
        logger.info(f"  {lang}: {count}")

    # Show sample pairs
    logger.info("\n=== Sample Extracted Pairs ===")
    for idx, row in output_df.head(5).iterrows():
        logger.info(f"\nPair {idx + 1}:")
        logger.info(f"  oare_id: {row['oare_id']}")
        logger.info(f"  Language: {row['translation_language']}")
        logger.info(f"  Pattern: {row['extraction_pattern']}")
        logger.info(f"  Transliteration: {row['transliteration'][:100]}...")
        logger.info(f"  Translation: {row['translation'][:100]}...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract translations from publications.csv and match to published_texts.csv"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw'),
        help='Path to data directory containing CSV files',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/publication_translations.csv'),
        help='Output CSV file path',
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Process only first N pages (for testing)',
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum translation length (default: 10)',
    )

    args = parser.parse_args()

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    process_publications(
        data_dir=args.data_dir,
        output_path=args.output,
        sample_size=args.sample,
        min_translation_length=args.min_length,
    )


if __name__ == "__main__":
    main()
