import os
from typing import Optional


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv_if_present(path: Optional[str] = None) -> Optional[str]:
    dotenv_path = path or os.path.join(os.getcwd(), '.env')
    if not os.path.exists(dotenv_path):
        return None

    with open(dotenv_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = _strip_wrapping_quotes(value.strip())
            if key and key not in os.environ:
                os.environ[key] = value
    return dotenv_path
