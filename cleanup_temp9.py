#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理临时文件"""

import subprocess
from pathlib import Path

# 删除临时文件
temp_files = [
    "cleanup_temp8.py",
]

for f in temp_files:
    p = Path(f)
    if p.exists():
        p.unlink()
        print(f"✓ 删除: {f}")

print("\n完成！")
