import re
from typing import List, Optional

def correct_oracle_template(template_string: str, user_strings: Optional[List[str]] = None) -> str:
    """
    Applies heuristic rules to correct a single, manually-generated oracle template string.
    """

    corrected_template = template_string

    # Digit (DG) - standalone numbers -> "<*>"
    corrected_template = re.sub(r'\b\d+\b', '<*>', corrected_template)

    # Boolean (BL) - true/false (case-insensitive) -> "<*>"
    corrected_template = re.sub(r'\b(true|false)\b', '<*>', corrected_template, flags=re.IGNORECASE)

    win_path = r'(?<![A-Za-z0-9._-])(?:[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+\\?)(?![A-Za-z0-9._-])'
    corrected_template = re.sub(win_path, '<*>', corrected_template)

    # Path String (PS) - simplified Unix-like paths (incl. ~/...) -> "<*>"
    # Use "not path-char" boundaries instead of \b around '/'
    PATH = r'(?<![A-Za-z0-9._-])(?:~?/)(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+/?(?![A-Za-z0-9._-])'
    corrected_template = re.sub(PATH, '<*>', corrected_template)

    # User-defined String (US) - treat provided strings as variables
    # Use token boundaries that work with non-word chars, e.g. "C++"
    if user_strings:
        unique_terms = sorted({s for s in user_strings if s}, key=len, reverse=True)
        escaped_terms = [re.escape(s) for s in unique_terms]
        user_regex = r'(?<!\w)(?:' + '|'.join(escaped_terms) + r')(?!\w)'
        corrected_template = re.sub(user_regex, '<*>', corrected_template, flags=re.IGNORECASE)

    MT_SEG = r'(?:(?<=^)|(?<=[\s,;:(){}\[\]"\'\']))[ ^,\s;:(){}\[\]"\'\']*<\*>[^,\s;:(){}\[\]"\'\']*(?=(?:$ |[\s, ;:(){}\[\]"\'\']))'
    corrected_template = re.sub(MT_SEG, '<*>', corrected_template)

    # Dot-separated Variables (DV) - collapse "<*>.<*>" chains to one "<*>"
    corrected_template = re.sub(r'(?:<\*>\s*\.\s*)+<\*>', '<*>', corrected_template)

    # Consecutive Variables (CV) - collapse "<*> <*> <*> ..." to one "<*>"
    corrected_template = re.sub(r'(?:<\*>\s*)+<\*>', '<*>', corrected_template)

    # Double Spaces (DS) - collapse runs of spaces/tabs, preserve newlines
    corrected_template = re.sub(r'[ \t]{2,}', ' ', corrected_template)

    return corrected_template.strip()
