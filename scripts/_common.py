"""Configuration and process helpers for the command-line entrypoints."""

import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("RS", "Med", "AD", "Sci", "Fin")


def path(value):
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else ROOT / expanded


def read_json(filename):
    with path(filename).open() as handle:
        return json.load(handle)


def require_paths(values):
    missing = [str(path(value)) for value in values if not path(value).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def environment():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env


def arguments(values):
    result = []
    for key, value in values.items():
        if value is not None:
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            result.extend(["--" + key, str(value)])
    return result


def run(command, env=None, dry_run=False):
    print(shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, env=env or environment(), check=True)
