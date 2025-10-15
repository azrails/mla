from .config import Config
from pathlib import Path
from loguru import logger
import zipfile
import shutil


def copytree(src: Path, dst: Path, use_symlinks=True) -> None:
    """
    Copy contents of `src` to `dst`. Unlike shutil.copytree, the dst dir can exist and will be merged.
    If src is a file, only that file will be copied. Optionally uses symlinks instead of copying.

    Args:
        src (Path): source directory
        dst (Path): destination directory
    """
    assert dst.is_dir()

    if src.is_file():
        dest_f = dst / src.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            dest_f.symlink_to(src)
        else:
            shutil.copyfile(src, dest_f)
        return

    for f in src.iterdir():
        dest_f = dst / f.name
        assert not dest_f.exists(), dest_f
        if use_symlinks:
            dest_f.symlink_to(f)
        elif f.is_dir():
            shutil.copytree(f, dest_f)
        else:
            shutil.copyfile(f, dest_f)


def clean_up_dataset(path: Path) -> None:
    for item in path.rglob("__MACOSX"):
        if item.is_dir():
            shutil.rmtree(item)
    for item in path.rglob(".DS_Store"):
        if item.is_file():
            item.unlink()


def preprocess_data(path: Path) -> None:
    for zip_file in path.rglob("*.zip"):
        out_dir = zip_file.with_suffix("")
        if out_dir.exists():
            logger.info(
                f"Skipping {zip_file} as an item with the same name already exists."
            )
            if out_dir.is_file() and out_dir.suffix != "":
                zip_file.unlink()
            continue

        logger.info(f"Extracting: {zip_file}")
        out_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(out_dir)

        # remove any unwanted files
        clean_up_dataset(out_dir)

        contents = list(out_dir.iterdir())

        # special case: the zip contains a single dir/file with the same name as the zip
        if len(contents) == 1 and contents[0].name == out_dir.name:
            sub_item = contents[0]
            # if it's a dir, move its contents to the parent and remove it
            if sub_item.is_dir():
                logger.info(f"Special handling (child is dir) enabled for: {zip_file}")
                for f in sub_item.rglob("*"):
                    shutil.move(f, out_dir)
                sub_item.rmdir()
            # if it's a file, rename it to the parent and remove the parent
            elif sub_item.is_file():
                logger.info(f"Special handling (child is file) enabled for: {zip_file}")
                sub_item_tmp = sub_item.rename(out_dir.with_suffix(".__tmp_rename"))
                out_dir.rmdir()
                sub_item_tmp.rename(out_dir)
        zip_file.unlink()


def prepare_workspase(cfg: Config) -> None:
    """Create agent workspase, extract and clean dataset files

    Args:
        cfg (Config): agent config
    """
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "data_dir").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "code").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(
        cfg.data_dir, cfg.workspace_dir / "data_dir", use_symlinks=not cfg.copy_data
    )
    if cfg.preprocess_data:
        preprocess_data(cfg.workspace_dir / "data_dir")
