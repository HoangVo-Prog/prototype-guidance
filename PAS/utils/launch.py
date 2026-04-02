import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


_RUN_NAME_ENV_VAR = 'PAS_RUN_NAME_OVERRIDE'
_BOOLEAN_STRINGS = {'1', 'true', 't', 'yes', 'y', 'on', '0', 'false', 'f', 'no', 'n', 'off'}
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def build_run_name(args):
    override = os.environ.get(_RUN_NAME_ENV_VAR)
    if override:
        return override
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    base_name = getattr(args, 'name', None) or getattr(args, 'model_variant', 'pas_v1')
    return f'{timestamp}_{base_name}'


def get_effective_wandb_run_name(args):
    return getattr(args, 'wandb_run_name', None) or getattr(args, 'run_name', None) or build_run_name(args)


def get_logs_dir():
    return Path(__file__).resolve().parents[1] / 'logs'


def build_nohup_log_path(args):
    run_name = get_effective_wandb_run_name(args)
    if _INVALID_FILENAME_CHARS.search(run_name):
        raise ValueError(
            'The W&B run name must not contain filesystem-invalid characters when --nohup is enabled: '
            f'{run_name!r}'
        )
    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f'{run_name}.log'


def strip_nohup_flag(argv):
    cleaned = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == '--nohup':
            next_index = index + 1
            if next_index < len(argv) and argv[next_index].strip().lower() in _BOOLEAN_STRINGS:
                skip_next = True
            continue
        if token.startswith('--nohup='):
            continue
        cleaned.append(token)
    return cleaned


def launch_with_nohup(argv, log_path, run_name_override=None, cwd=None):
    normalized_argv = list(strip_nohup_flag(argv))
    if normalized_argv:
        normalized_argv[0] = str(Path(normalized_argv[0]).resolve())
    child_argv = [sys.executable, *normalized_argv]
    child_env = os.environ.copy()
    if run_name_override:
        child_env[_RUN_NAME_ENV_VAR] = run_name_override

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(log_path, 'ab')
    try:
        if os.name == 'posix' and shutil.which('nohup'):
            command = ['nohup', *child_argv]
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        else:
            creationflags = 0
            if os.name == 'nt':
                creationflags = (
                    getattr(subprocess, 'DETACHED_PROCESS', 0)
                    | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
                )
            process = subprocess.Popen(
                child_argv,
                cwd=cwd,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != 'nt'),
                creationflags=creationflags,
            )
    finally:
        stdout_handle.close()
    return process.pid
