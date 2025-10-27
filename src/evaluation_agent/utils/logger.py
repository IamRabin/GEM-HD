import logging
from pathlib import Path

def setup_logger(name: str = "GEMHD", log_dir: str | Path | None = None):
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent double logging if called multiple times
    if logger.handlers:
        return logger

    # Create console handler with emoji formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("🔹 %(asctime)s — %(levelname)s — %(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional: also log to a file
    if log_dir:
        log_dir = Path(log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "evaluation_agent.log")
        fh.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger