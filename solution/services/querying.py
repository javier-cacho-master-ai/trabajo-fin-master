import re

def sanitize_adql(raw_query: str) -> str:
    """Removes SQL comments and compresses whitespace into single spaces."""
    # 1. Remove block comments (/* ... */)
    # The re.DOTALL flag allows '.' to match newline characters
    no_blocks = re.sub(r"/\*.*?\*/", "", raw_query, flags=re.DOTALL)

    # 2. Remove inline comments (-- ...)
    no_inlines = re.sub(r"--.*", "", no_blocks)

    # 3. Compress remaining whitespace (tabs, newlines, multiple spaces)
    return re.sub(r"\s+", " ", no_inlines).strip()
