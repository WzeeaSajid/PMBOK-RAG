"""Launch the application with per-process model and context settings."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Value must be a positive integer.')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Installed Ollama model name.')
    parser.add_argument('--context', type=positive_integer, help='Ollama context window.')
    parser.add_argument('--evidence-budget', type=positive_integer, help='Source context budget.')
    parser.add_argument('--max-output', type=positive_integer, help='Maximum generated tokens.')
    parser.add_argument('--keep-alive', help='Ollama retention duration, or -1 to keep loaded.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    executable = Path(sys.executable).with_name('chainlit.exe' if os.name == 'nt' else 'chainlit')
    if not executable.is_file():
        parser.error('Chainlit is not installed in this environment. Install requirements.txt first.')
    environment = os.environ.copy()
    overrides = {
        'PMBOK_LLM_MODEL': args.model,
        'PMBOK_LLM_CONTEXT': args.context,
        'PMBOK_CONTEXT_TOKENS': args.evidence_budget,
        'PMBOK_LLM_MAX_TOKENS': args.max_output,
        'PMBOK_KEEP_ALIVE': args.keep_alive,
    }
    environment.update({key: str(value) for key, value in overrides.items() if value is not None})
    command = [str(executable), 'run', str(root / 'src' / 'chainlit_app.py')]
    try:
        return subprocess.call(command, cwd=root, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
