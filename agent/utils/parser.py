from .config import Config
from loguru import logger


def task_parser(cfg: Config):
    """Parse task text from file or cli argumnet to agent prompt format

    Args:
        cfg (Config): agent config
    """
    if cfg.desc_file is not None:
        with open(cfg.desc_file) as f:
            return f.read().strip() + "\n"
    else:
        description = {}
        description["Task description:"] = cfg.desc.strip()
        if cfg.eval != None:
            description["Task evaluation"] = cfg.eval.strip()
    description = ["#" + key + "\n" + val for key, val in description.items()]
    return "\n".join(description)
