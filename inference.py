"""Root inference entrypoint for hackathon validation.

This file intentionally delegates all logic to files.inference.main,
and contains no duplicate inference logic.
"""

from files.inference import main


if __name__ == "__main__":
    main()
