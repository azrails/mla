from .utils import (
    task_parser,
    load_cfg,
    setup_logger,
    Config,
    prepare_workspase,
    get_model,
)
from .agent import run_pipeline
from loguru import logger
import atexit
import shutil


def cleanup_workspase(cfg: Config):
    shutil.rmtree(cfg.workspace_dir.parent)

def check_submission(cfg: Config):
    if not (cfg.workspace_dir / "submission" / "submission.csv").exists():
        return False
    return True

def run():
    cfg = load_cfg()
    setup_logger(cfg)
    logger.info(f'RUN AGENT "{cfg.exp_name}"')
    task_description = task_parser(cfg)
    prepare_workspase(cfg)
    atexit.register(cleanup_workspase, cfg)
    expert_model = get_model(cfg, model_type="expert")
    feedback_model = get_model(cfg, model_type="feedback")
    code_model = get_model(cfg, model_type="code")

    bug_issue = False
    while not check_submission(cfg):
        logger.info(run_pipeline(code_model, expert_model, feedback_model, cfg, task_description))
        if bug_issue is False:
            task_description + f'\nYou a currently generate a BUG solution without creating a submission.txt file in {cfg.workspace_dir / "submission" / "submission.csv"}\n'


if __name__ == "__main__":
    run()
