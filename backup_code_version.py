from datetime import datetime
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
VERSION_DIR = ROOT / "code_versions"
FILES_TO_BACKUP = [
    "01_generate_collision_data.py",
    "02_train_transformer.py",
]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "manual_backup"
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = VERSION_DIR / f"{timestamp}_{safe_tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    for name in FILES_TO_BACKUP:
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    readme = out_dir / "README.txt"
    readme.write_text(
        "版本说明：\\n"
        f"- 备份标签：{tag}\\n"
        f"- 备份时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        "修改目的：\\n"
        "- 待填写。\\n\\n"
        "主要改动：\\n"
        "- 待填写。\\n\\n"
        "预期观察指标：\\n"
        "- D_MAE\\n"
        "- T_MAE\\n"
        "- 危险召回率\\n"
        "- 误报率 / 漏报率\\n\\n"
        "运行结果：\\n"
        "- 待填写。\\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
