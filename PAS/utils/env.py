import os
from typing import Optional


WANDB_API_KEY_NAME = 'WANDB_API_KEY'


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


def load_kaggle_secret_if_present(secret_name: str = WANDB_API_KEY_NAME) -> Optional[str]:
    if os.environ.get(secret_name):
        return secret_name

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None

    try:
        secret_value = UserSecretsClient().get_secret(secret_name)
    except Exception:
        return None

    if secret_value:
        os.environ[secret_name] = secret_value
        return secret_name
    return None


def load_runtime_environment(dotenv_path: Optional[str] = None) -> None:
    load_dotenv_if_present(dotenv_path)
    if not os.environ.get(WANDB_API_KEY_NAME):
        load_kaggle_secret_if_present(WANDB_API_KEY_NAME)
