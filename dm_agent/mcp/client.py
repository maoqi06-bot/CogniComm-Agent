"""MCP 客户端 - 修复了协议握手与 JSON 过滤逻辑"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
from threading import Thread, Lock
from queue import Queue, Empty


class MCPClient:
    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.tools: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._message_id = 0
        self._stdout_queue: Queue = Queue()
        self._running = False

    @staticmethod
    def _project_root() -> str:
        return str(Path(__file__).resolve().parents[2])

    @classmethod
    def _expand_value(cls, value: str, env: Dict[str, str]) -> str:
        replacements = {
            "PROJECT_ROOT": cls._project_root(),
            "PYTHON": sys.executable,
            "PYTHON_EXECUTABLE": sys.executable,
        }

        expanded = value
        for key, replacement in replacements.items():
            expanded = expanded.replace(f"${{{key}}}", replacement)

        for key, replacement in env.items():
            expanded = expanded.replace(f"${{{key}}}", replacement)

        return os.path.expandvars(os.path.expanduser(expanded))

    @staticmethod
    def _resolve_command(command: str) -> str:
        return shutil.which(command) or command

    def start(self) -> bool:
        try:
            process_env = os.environ.copy()
            process_env.setdefault("PROJECT_ROOT", self._project_root())
            process_env.setdefault("PYTHON", sys.executable)
            process_env.setdefault("PYTHON_EXECUTABLE", sys.executable)
            if self.env:
                process_env.update({
                    key: self._expand_value(str(value), process_env)
                    for key, value in self.env.items()
                })

            # Windows 兼容性处理
            is_windows = sys.platform == 'win32'
            command = self._resolve_command(self._expand_value(self.command, process_env))
            args = [self._expand_value(str(arg), process_env) for arg in self.args]
            full_command = [command] + args

            self.process = subprocess.Popen(
                full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,  # 修改点1：直接把子进程报错打印到当前 main.py 窗口
                env=process_env,
                cwd=self._project_root(),
                shell=False,
                text=True,
                bufsize=0,  # 修改点2：禁用缓冲，哪怕只有半行报错也立刻显示
                encoding='utf-8'  # 修改点3：强制 UTF-8，防止 Windows 默认 GBK 导致解析 JSON 失败
            )

            self._running = True
            self._stdout_thread = Thread(target=self._read_stdout, daemon=True)
            self._stdout_thread.start()

            # 核心修复：执行完整的初始化握手
            if not self._initialize_handshake():
                self.stop()
                return False

            print(f"✅ MCP 服务器 '{self.name}' 启动成功，提供 {len(self.tools)} 个工具")
            return True

        except Exception as e:
            print(f"❌ 启动 MCP 服务器 '{self.name}' 失败: {e}")
            return False

    def _read_stdout(self) -> None:
        if not self.process or not self.process.stdout:
            return
        while self._running:
            line = self.process.stdout.readline()
            if not line: break
            # 过滤非 JSON 行（防止 npx 的警告信息干扰）
            clean_line = line.strip()
            if clean_line.startswith('{'):
                self._stdout_queue.put(clean_line)
            else:
                # 可以在这里打印非协议日志以便调试
                # print(f"DEBUG [{self.name}]: {clean_line}")
                pass

    def _send_message(self, method: str, params: Optional[Dict[str, Any]] = None, is_notification: bool = False) -> \
    Optional[Dict[str, Any]]:
        if not self.process or not self.process.stdin:
            return None

        with self._lock:
            self._message_id += 1
            message = {
                "jsonrpc": "2.0",
                "method": method,
            }
            if not is_notification:
                message["id"] = self._message_id
            if params:
                message["params"] = params

            try:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()

                if is_notification:
                    return {"status": "sent"}

                # 等待响应逻辑
                timeout_limit = 120.0  # 延长至 10 秒
                start_time = time.time()
                while (time.time() - start_time) < timeout_limit:
                    try:
                        response_line = self._stdout_queue.get(timeout=0.1)
                        response = json.loads(response_line)

                        if response.get("id") == self._message_id:
                            if "error" in response:
                                print(f"❌ MCP [{self.name}] 错误: {response['error']}")
                                return None
                            return response.get("result")

                        # 不是当前请求的响应，放回队列供后续处理
                        self._stdout_queue.put(response_line)
                    except Empty:
                        continue
                    except Exception:
                        continue

                print(f"⚠️ MCP [{self.name}] 响应超时 (Method: {method})")
                return None
            except Exception as e:
                print(f"❌ 发送消息到 [{self.name}] 失败: {e}")
                return None

    def _initialize_handshake(self) -> bool:
        """执行 MCP 标准初始化流程"""
        # 1. 发送 initialize
        init_result = self._send_message("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "dm-code-agent", "version": "1.1.0"}
        })
        if not init_result: return False

        # 2. 发送 notifications/initialized (关键补丁)
        self._send_message("notifications/initialized", is_notification=True)

        # 3. 获取工具列表
        tools_result = self._send_message("tools/list")
        if tools_result and "tools" in tools_result:
            self.tools = tools_result["tools"]
            return True
        return False

    def stop(self) -> None:
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                if self.process: self.process.kill()
            self.process = None
        print(f"🛑 MCP 服务器 '{self.name}' 已停止")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        result = self._send_message("tools/call", {"name": tool_name, "arguments": arguments})
        if result and "content" in result:
            content = result["content"]
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", str(content[0]))
            return str(content)
        return None

    def get_tools(self) -> List[Dict[str, Any]]:
        return self.tools.copy()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
