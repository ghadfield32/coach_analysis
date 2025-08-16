"""
Centralized name normalization utilities for salary_nba_data_pull.

This module provides a single source of truth for player name normalization
to ensure consistent matching between NBA API and Basketball Reference data.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Dict, Any

# Common suffix patterns to remove
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", flags=re.I)

# Conservative mapping of dotted initials to compact forms
_INITIAL_MAP = {
    "a.j.": "aj", "b.j.": "bj", "c.j.": "cj", "d.j.": "dj", "e.j.": "ej",
    "g.g.": "gg", "j.j.": "jj", "k.j.": "kj", "p.j.": "pj", "r.j.": "rj",
    "t.j.": "tj", "m.j.": "mj", "w.j.": "wj",
}

def normalize_name(name: str) -> str:
    """
    Create a deterministic key from a player name.
    
    Process:
    1. Trim and lowercase
    2. Fold dotted initials (A.J. -> aj)
    3. Remove common suffixes (jr/sr/ii/iii/iv/v)
    4. Unicode NFKD normalization + remove combining marks
    5. Keep only alphanumeric and spaces
    6. Collapse multiple spaces
    
    Args:
        name: Raw player name string
        
    Returns:
        Normalized key string for matching
        
    Examples:
        >>> normalize_name("Luka Dončić")
        'luka doncic'
        >>> normalize_name("A.J. Green Jr.")
        'aj green'
        >>> normalize_name("Dennis Schröder")
        'dennis schroder'
    """
    if not name:
        return ""
    
    # Step 1: Basic cleanup
    s = str(name).strip().lower()
    
    # Step 2: Fold dotted initials
    for pattern, replacement in _INITIAL_MAP.items():
        s = s.replace(pattern, replacement)
    
    # Step 3: Remove suffixes
    s = _SUFFIX_RE.sub("", s).strip()
    
    # Step 4: Remove obvious punctuation used on some sites
    s = re.sub(r"[(),]", " ", s)
    
    # Step 5: Unicode normalization (NFKD decomposes diacritics)
    s = unicodedata.normalize("NFKD", s)
    # Remove combining marks (diacritics)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    
    # Step 6: Keep only alphanumeric and spaces
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    # Step 7: Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

def validate_name_encoding(df: 'pd.DataFrame', season: str, debug: bool = False) -> bool:
    """
    Validate that expected Unicode names appear correctly in the DataFrame.
    
    Args:
        df: DataFrame with 'Player' column
        season: Season string for logging
        debug: Whether to print detailed diagnostics
        
    Returns:
        True if all expected names are found, False otherwise
        
    Raises:
        AssertionError: If critical encoding issues are detected
    """
    if 'Player' not in df.columns:
        if debug:
            print(f"[validate_name_encoding] {season}: No 'Player' column found")
        return False
    
    # Test cases with diacritics that should appear correctly
    test_names = [
        "Luka Dončić",
        "Nikola Jokić", 
        "Dennis Schröder",
        "Bojan Bogdanović"
    ]
    
    found_names = []
    missing_names = []
    
    for test_name in test_names:
        if (df['Player'] == test_name).any():
            found_names.append(test_name)
        else:
            missing_names.append(test_name)
    
    if debug:
        print(f"[validate_name_encoding] {season}: Found {len(found_names)}/{len(test_names)} expected names")
        if missing_names:
            print(f"  Missing: {missing_names}")
            # Show some actual names for debugging
            sample_names = df['Player'].dropna().head(10).tolist()
            print(f"  Sample actual names: {sample_names}")
    
    # If we're missing critical names, this indicates encoding issues
    if len(missing_names) > 2:  # Allow for some variation
        raise AssertionError(
            f"[validate_name_encoding] {season}: Critical encoding issues detected. "
            f"Missing {len(missing_names)} expected Unicode names: {missing_names}"
        )
    
    return len(missing_names) == 0 
