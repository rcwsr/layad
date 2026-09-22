"""Entry point for `python -m layad`.

The LaunchAgent invokes layad this way — `[sys.executable, "-m", "layad", "serve"]`
resolves without depending on PATH, which launchd does not inherit from your shell.
"""

from .cli import main

if __name__ == "__main__":
    main()
