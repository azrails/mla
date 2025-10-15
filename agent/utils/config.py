from omegaconf import OmegaConf
from pathlib import Path
from typing import Hashable, cast
from dataclasses import dataclass
import uuid
import rich
from rich.syntax import Syntax
from loguru import logger


@dataclass
class StageConfig:
    model: str
    temp: float


@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool

    code: StageConfig
    feedback: StageConfig
    expert: StageConfig


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    desc: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def prep_cfg(cfg: Config) -> Config:
    """Validate and freeze config

    Args:
        cfg (Config): agent config
    """
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.desc is None:
        raise ValueError(
            "You must provide either a description of the task goal (`desc=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    cfg.exp_name = cfg.exp_name or str(uuid.uuid4())

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)

    # validate the config and freeze fields
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    OmegaConf.set_readonly(cfg, True)
    return cast(Config, cfg)


def log_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    cfg = prep_cfg(_load_cfg(path))
    log_cfg(cfg)
    return cfg
