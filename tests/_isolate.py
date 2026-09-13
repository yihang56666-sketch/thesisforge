# -*- coding: utf-8 -*-
"""测试隔离层：必须在任何 `import app.*` 之前先导入本模块。

做两件事：
1. 把仓库根加进 sys.path，使 `python -m unittest discover -s tests` 无需安装；
2. 把 THESISFORGE_DATA_DIR 指向一次性临时目录 —— app.config 在导入时就把
   DATA_DIR 定死了，所以这一步必须早于任何 app 模块导入，否则测试会往
   用户真实的 data/ 里写数据集和实验记录。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_TMP = Path(tempfile.mkdtemp(prefix="thesisforge-test-"))
os.environ["THESISFORGE_DATA_DIR"] = str(TEST_TMP)


def scratch(name: str) -> Path:
    """给单个用例一个互不干扰的子目录。"""
    p = TEST_TMP / name
    p.mkdir(parents=True, exist_ok=True)
    return p
