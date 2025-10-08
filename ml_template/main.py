import sys
from src.cli import run_cli

if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        sys.exit(0)
