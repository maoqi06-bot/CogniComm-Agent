"""代码执行工具"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import _require_str


def _truncate_output(text: str, *, max_chars: int | None = None) -> str:
    if max_chars is None:
        max_chars = int(os.getenv("TOOL_MAX_OUTPUT_CHARS", "12000"))
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        f"{text[:half]}\n\n"
        f"... <truncated {omitted} chars of command output> ...\n\n"
        f"{text[-half:]}"
    )


def _tool_timeout(arguments: Dict[str, Any]) -> float:
    value = arguments.get("timeout", os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "120"))
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        timeout = 120.0
    return max(1.0, timeout)


def run_python(arguments: Dict[str, Any]) -> str:
    """运行 Python 代码或脚本"""
    code = arguments.get("code")
    path_value = arguments.get("path")

    if isinstance(code, str) and code.strip():
        command = [sys.executable, "-u", "-c", code]
    elif isinstance(path_value, str) and path_value.strip():
        command = [sys.executable, "-u", str(Path(path_value))]
        extra_args = arguments.get("args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args)
        elif isinstance(extra_args, str) and extra_args.strip():
            command.extend(shlex.split(extra_args))
        elif extra_args is not None:
            raise ValueError("工具参数 'args' 必须是字符串或字符串列表。")
    else:
        raise ValueError("run_python 工具需要 'code' 或 'path' 参数。")

    # 使用显式的 stdout/stderr 管道，并指定 utf-8 编码，忽略解码错误
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=_tool_timeout(arguments),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return "\n".join(
            segment
            for segment in [
                _truncate_output(output.strip()) if output else "",
                f"stderr:\n{_truncate_output(stderr.strip())}" if stderr else "",
                f"returncode: timeout after {_tool_timeout(arguments):.0f}s",
            ]
            if segment
        )
    segments: List[str] = []
    if result.stdout:
        segments.append(_truncate_output(result.stdout.strip()))
    if result.stderr:
        segments.append(f"stderr:\n{_truncate_output(result.stderr.strip())}")
    segments.append(f"returncode: {result.returncode}")
    return "\n".join(segment for segment in segments if segment).strip() or "returncode: 0"


def run_shell(arguments: Dict[str, Any]) -> str:
    """运行 Shell 命令"""
    command = _require_str(arguments, "command")
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=_tool_timeout(arguments),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return "\n".join(
            segment
            for segment in [
                _truncate_output(output.strip()) if output else "",
                f"stderr:\n{_truncate_output(stderr.strip())}" if stderr else "",
                f"returncode: timeout after {_tool_timeout(arguments):.0f}s",
            ]
            if segment
        )
    segments: List[str] = []
    if result.stdout:
        segments.append(_truncate_output(result.stdout.strip()))
    if result.stderr:
        segments.append(f"stderr:\n{_truncate_output(result.stderr.strip())}")
    segments.append(f"returncode: {result.returncode}")
    return "\n".join(segment for segment in segments if segment).strip() or "returncode: 0"


def run_tests(arguments: Dict[str, Any]) -> str:
    """运行 Python 测试套件（支持 pytest 和 unittest）"""
    test_path = arguments.get("test_path", ".")
    framework = arguments.get("framework", "pytest")
    verbose = arguments.get("verbose", False)

    if not isinstance(test_path, str):
        raise ValueError("test_path 必须是字符串。")

    if framework not in ["pytest", "unittest"]:
        raise ValueError("framework 必须是 'pytest' 或 'unittest'。")

    path = Path(test_path)
    if not path.exists():
        return f"测试路径 {path} 不存在。"

    if framework == "pytest":
        command = [sys.executable, "-m", "pytest"]
        if verbose:
            command.append("-v")
        command.append(str(path))
    else:  # unittest
        command = [sys.executable, "-m", "unittest"]
        if verbose:
            command.append("-v")
        if path.is_file():
            # 转换为模块路径
            module_path = str(path).replace("/", ".").replace("\\", ".").replace(".py", "")
            command.append(module_path)
        else:
            command.extend(["discover", "-s", str(path)])

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=_tool_timeout(arguments),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return "\n".join(
            segment
            for segment in [
                _truncate_output(output.strip()) if output else "",
                f"stderr:\n{_truncate_output(stderr.strip())}" if stderr else "",
                f"returncode: timeout after {_tool_timeout(arguments):.0f}s",
            ]
            if segment
        )
    segments: List[str] = []

    if result.stdout:
        segments.append(_truncate_output(result.stdout.strip()))
    if result.stderr:
        segments.append(f"stderr:\n{_truncate_output(result.stderr.strip())}")
    segments.append(f"returncode: {result.returncode}")

    output = "\n".join(segment for segment in segments if segment).strip()
    return output if output else "returncode: 0"


def run_linter(arguments: Dict[str, Any]) -> str:
    """运行代码检查工具（支持 pylint、flake8、mypy、black）"""
    path_value = _require_str(arguments, "path")
    tool = arguments.get("tool", "flake8")

    if tool not in ["pylint", "flake8", "mypy", "black"]:
        raise ValueError("tool 必须是 'pylint'、'flake8'、'mypy' 或 'black' 之一。")

    path = Path(path_value)
    if not path.exists():
        return f"路径 {path} 不存在。"

    if tool == "black":
        # black 用于格式化，添加 --check 只检查不修改
        command = [sys.executable, "-m", tool, "--check", str(path)]
    else:
        command = [sys.executable, "-m", tool, str(path)]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=_tool_timeout(arguments),
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return "\n".join(
            segment
            for segment in [
                _truncate_output(output.strip()) if output else "",
                f"stderr:\n{_truncate_output(stderr.strip())}" if stderr else "",
                f"returncode: timeout after {_tool_timeout(arguments):.0f}s",
            ]
            if segment
        )
    segments: List[str] = []

    if result.stdout:
        segments.append(_truncate_output(result.stdout.strip()))
    if result.stderr:
        segments.append(f"stderr:\n{_truncate_output(result.stderr.strip())}")
    segments.append(f"returncode: {result.returncode}")

    output = "\n".join(segment for segment in segments if segment).strip()
    if output:
        return output
    return f"{tool} 检查通过，未发现问题。"
