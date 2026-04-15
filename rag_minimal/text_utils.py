#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文本处理公共工具。"""
from __future__ import annotations

import re
from typing import List, Set

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def unique_tokens(text: str) -> Set[str]:
    return set(tokenize(text))
