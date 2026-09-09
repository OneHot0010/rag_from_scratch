"""表格类 loader 的共享工具:类型规范化、表头处理、行转文本。

Excel 与 CSV 面临的很多"坑"是同源的(类型失真、空表头、宽表、空行),
把这些共性逻辑收敛到一处,避免在两个 loader 里重复实现、各自跑偏。
"""
from __future__ import annotations

import datetime as _dt
from decimal import Decimal
from typing import Any, List, Optional, Sequence

# 单个字段过长时截断,避免超长文本稀释 embedding 检索信号。
MAX_FIELD_CHARS = 500


def normalize_value(v: Any) -> str:
    """把单元格值规范化为稳定、无失真的字符串。

    处理要点(工业级常见坑):
      - None / 空       -> ""(交由上层决定是否跳过)
      - 日期/时间       -> ISO 格式,避免变成 Excel 序列号或本地化字符串
      - bool            -> 显式 true/false(不要变成 1/0)
      - float 整数值    -> 去掉多余的 .0(如 10.0 -> "10")
      - Decimal / 大数  -> 用普通十进制,避免科学计数法
      - 其他            -> str() 后 strip
    """
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (_dt.datetime,)):
        # 若为纯日期(时分秒为0)只保留日期部分
        if v.hour == v.minute == v.second == 0 and v.microsecond == 0:
            return v.date().isoformat()
        return v.isoformat(sep=" ")
    if isinstance(v, _dt.date):
        return v.isoformat()
    if isinstance(v, _dt.time):
        return v.isoformat()
    if isinstance(v, Decimal):
        return format(v.normalize(), "f")
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        # 抑制浮点噪声(如 0.30000000000000004),保留合理精度
        return repr(round(v, 10)).rstrip("0").rstrip(".") if "." in repr(v) else repr(v)
    s = str(v).strip()
    return s


def clean_headers(raw_headers: Sequence[Any]) -> List[str]:
    """规范化表头:空表头补 col{i},去重(重名列加后缀)。"""
    headers: List[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(raw_headers):
        name = normalize_value(h) or f"col{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        headers.append(name)
    return headers


def merge_header_rows(header_rows: Sequence[Sequence[Any]]) -> List[str]:
    """合并多级表头:逐列把各层非空文本用 '-' 连接。

    例如两行表头 [["销售", None], ["Q1", "Q2"]] -> ["销售-Q1", "销售-Q2"]。
    仅在调用方显式指定 header_rows>1 时使用。
    """
    if not header_rows:
        return []
    width = max(len(r) for r in header_rows)
    merged: List[str] = []
    for col in range(width):
        parts: List[str] = []
        for row in header_rows:
            val = normalize_value(row[col]) if col < len(row) else ""
            if val and (not parts or parts[-1] != val):
                parts.append(val)
        merged.append("-".join(parts))
    return clean_headers(merged)


def row_to_text(headers: Sequence[str], values: Sequence[Any],
                sheet: Optional[str] = None) -> str:
    """把一行数据拼成 "列名: 值；..." 的自然语言式文本。

    - 跳过空值,避免噪声;
    - 单字段超长截断;
    - 可选把 sheet/表名作为语义锚点前置,缓解宽表/多 sheet 同名列歧义。
    """
    parts: List[str] = []
    for i, v in enumerate(values or ()):
        name = headers[i] if i < len(headers) else f"col{i + 1}"
        text = normalize_value(v)
        if not text:
            continue
        if len(text) > MAX_FIELD_CHARS:
            text = text[:MAX_FIELD_CHARS] + "…"
        parts.append(f"{name}: {text}")
    body = "；".join(parts)
    if sheet and body:
        return f"[表: {sheet}] {body}"
    return body


def is_blank_row(values: Sequence[Any]) -> bool:
    """整行为空(全 None/空串)则视为空行,跳过。"""
    return all(normalize_value(v) == "" for v in (values or ()))
