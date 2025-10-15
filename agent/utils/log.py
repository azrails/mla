from loguru import logger
from .config import Config
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(cfg: Config) -> None:
    """Setup logger module

    Args:
        cfg (Config): agent config
    """
    logger.remove()
    log_path = Path(cfg.log_dir).resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_path / "linguainterpreter.log",
        level=cfg.log_level,
        format="[{time:YYYY-MM-DD HH:mm:ss}] >> {level} >> {message}",
    )

    logger.add(
        RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_level=True,
        ),
        level=cfg.log_level,
    )
