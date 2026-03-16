from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(text)]
