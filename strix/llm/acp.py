import asyncio
import contextlib
import json
import shlex
import sys
import time
from asyncio.subprocess import PIPE, Process
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from strix.config import Config


class ACPError(Exception):
    pass


class ACPClient:
    def __init__(
        self,
        agent: str,
        timeout: int,
        cwd: str | None = None,
        command: str | None = None,
        permission: str | None = None,
    ):
        self.agent = agent
        self.timeout = timeout
        self.cwd = cwd or Config.get("strix_acp_cwd") or str(Path.cwd())
        self.command = command or Config.get("strix_acp_command") or self._default_command(agent)
        self.permission = permission or Config.get("strix_acp_permission") or "allow_once"

        self._process: Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._updates: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._session_id: str | None = None
        self._auth_methods: list[dict[str, Any]] = []
        self._config_options: list[dict[str, Any]] = []
        self._debug_log_path = Config.get("strix_acp_debug_log")
        self._idle_timeout = self._float_config("strix_acp_idle_timeout", 60.0)
        self._closed = False

    def _default_command(self, agent: str) -> str:
        if agent == "codex":
            return "codex-acp"
        return agent

    def _float_config(self, name: str, default: float) -> float:
        value = Config.get(name)
        if value is None:
            return default
        with contextlib.suppress(ValueError, TypeError):
            return float(value)
        return default

    async def start(self) -> None:
        if self._process is not None:
            return

        argv = shlex.split(self.command)
        if not argv:
            raise ACPError("STRIX_ACP_COMMAND resolved to an empty command")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE,
                cwd=self.cwd if Path(self.cwd).exists() else None,
            )
        except FileNotFoundError as e:
            raise ACPError(
                f"ACP command '{argv[0]}' was not found. "
                "Install codex-acp or set STRIX_ACP_COMMAND."
            ) from e

        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        await self._initialize()
        await self._authenticate_if_available()
        await self._new_session()

    async def close(self) -> None:
        self._closed = True
        if self._session_id:
            with contextlib.suppress(Exception):
                await self.request("session/close", {"sessionId": self._session_id})

        for task in (self._reader_task, self._stderr_task):
            if task and not task.done():
                task.cancel()

        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

    async def _initialize(self) -> None:
        result = await self.request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": "strix",
                    "title": "Strix",
                    "version": "0.8.3",
                },
            },
        )
        if isinstance(result, dict):
            auth_methods = result.get("authMethods")
            if isinstance(auth_methods, list):
                self._auth_methods = [m for m in auth_methods if isinstance(m, dict)]

    async def _authenticate_if_available(self) -> None:
        method_id = self._select_auth_method_id()
        if not method_id:
            return
        with contextlib.suppress(Exception):
            await self.request("authenticate", {"methodId": method_id})

    def _select_auth_method_id(self) -> str | None:
        if not self._auth_methods:
            return None

        configured = Config.get("strix_acp_auth_method")
        if configured:
            for method in self._auth_methods:
                if method.get("id") == configured:
                    return str(method["id"])

        preferred = ("chatgpt", "codex-api-key", "openai-api-key")
        for preferred_id in preferred:
            for method in self._auth_methods:
                if method.get("id") == preferred_id:
                    return preferred_id

        method_id = self._auth_methods[0].get("id")
        return str(method_id) if method_id else None

    async def _new_session(self) -> None:
        result = await self.request(
            "session/new",
            {
                "cwd": str(Path(self.cwd).resolve()),
                "mcpServers": self._mcp_servers(),
            },
        )
        if not isinstance(result, dict) or not result.get("sessionId"):
            raise ACPError(f"ACP session/new returned an invalid response: {result!r}")
        self._session_id = str(result["sessionId"])
        self._set_config_options(result.get("configOptions"))
        await self._configure_session()

    def _set_config_options(self, options: Any) -> None:
        if isinstance(options, list):
            self._config_options = [option for option in options if isinstance(option, dict)]

    def _mcp_servers(self) -> list[dict[str, Any]]:
        enabled = (Config.get("strix_acp_enable_mcp") or "true").lower()
        if enabled in {"0", "false", "no", "off"}:
            return []

        return [
            {
                "name": "strix",
                "command": sys.executable,
                "args": ["-m", "strix.llm.mcp_server"],
                "env": [
                    {"name": "STRIX_SANDBOX_MODE", "value": "false"},
                    {"name": "STRIX_DISABLE_BROWSER", "value": "true"},
                ],
            }
        ]

    async def _configure_session(self) -> None:
        if not self._session_id:
            return

        model = Config.get("strix_acp_model")
        reasoning = Config.get("strix_acp_reasoning_effort") or Config.get(
            "strix_reasoning_effort"
        )

        if model:
            await self._set_config_option(category="model", value=model)
        if reasoning:
            await self._set_config_option(category="thought_level", value=reasoning)

    async def _set_config_option(self, category: str, value: str) -> None:
        if not self._session_id:
            return

        option = self._find_config_option(category)
        if not option:
            return

        config_id = option.get("id")
        option_value = self._resolve_option_value(option, value)
        if not config_id or not option_value:
            return

        with contextlib.suppress(Exception):
            result = await self.request(
                "session/set_config_option",
                {
                    "sessionId": self._session_id,
                    "configId": str(config_id),
                    "value": option_value,
                },
            )
            if isinstance(result, dict):
                options = result.get("configOptions")
                if options:
                    self._set_config_options(options)

    def _find_config_option(self, category: str) -> dict[str, Any] | None:
        for option in self._config_options:
            if option.get("category") == category:
                return option
        for option in self._config_options:
            option_id = str(option.get("id") or "").lower()
            if category == "model" and "model" in option_id:
                return option
            if category == "thought_level" and any(
                part in option_id for part in ("reason", "thought", "effort")
            ):
                return option
        return None

    def _resolve_option_value(self, option: dict[str, Any], wanted: str) -> str | None:
        wanted_normalized = wanted.lower().replace("_", "-")
        options = option.get("options")
        if not isinstance(options, list):
            return wanted

        for candidate in options:
            if not isinstance(candidate, dict):
                continue
            value = str(candidate.get("value") or "")
            name = str(candidate.get("name") or "")
            if value == wanted or value.lower() == wanted_normalized:
                return value
            if name == wanted or name.lower().replace("_", "-") == wanted_normalized:
                return value

        return None

    async def prompt(self, text: str) -> AsyncIterator[dict[str, Any]]:
        await self.start()
        if not self._session_id:
            raise ACPError("ACP session is not initialized")

        request_id = await self.send_request(
            "session/prompt",
            {
                "sessionId": self._session_id,
                "prompt": [{"type": "text", "text": text}],
            },
        )
        pending = self._pending[request_id]
        last_update = time.monotonic()
        active_tools: dict[str, dict[str, Any]] = {}

        while not pending.done():
            try:
                update = await asyncio.wait_for(self._updates.get(), timeout=0.5)
            except TimeoutError:
                if (
                    active_tools
                    and self._idle_timeout > 0
                    and time.monotonic() - last_update > self._idle_timeout
                ):
                    await self._cancel_idle_turn(active_tools)
                continue
            last_update = time.monotonic()
            self._track_active_tool(active_tools, update)
            yield update

        while not self._updates.empty():
            yield self._updates.get_nowait()

        result = pending.result()
        yield {"sessionUpdate": "turn_complete", "result": result}

    def _track_active_tool(
        self, active_tools: dict[str, dict[str, Any]], update: dict[str, Any]
    ) -> None:
        update_type = update.get("sessionUpdate")
        tool_call_id = str(update.get("toolCallId") or "")
        if not tool_call_id:
            return
        if update_type == "tool_call":
            active_tools[tool_call_id] = update
        elif update_type == "tool_call_update":
            status = str(update.get("status") or "")
            if status in {"completed", "failed", "cancelled"}:
                active_tools.pop(tool_call_id, None)

    async def _cancel_idle_turn(self, active_tools: dict[str, dict[str, Any]]) -> None:
        with contextlib.suppress(Exception):
            await self.cancel()

        tool_hint = ""
        if active_tools:
            tool_id, tool = next(reversed(active_tools.items()))
            title = tool.get("title") or tool.get("kind") or "ACP tool"
            tool_hint = f" Last active tool: {title} ({tool_id})."
        debug_hint = (
            f" See STRIX_ACP_DEBUG_LOG at {self._debug_log_path}."
            if self._debug_log_path
            else ""
        )
        raise ACPError(
            f"ACP turn produced no events for {self._idle_timeout:g}s and was cancelled."
            f"{tool_hint}{debug_hint}"
        )

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request_id = await self.send_request(method, params or {})
        future = self._pending[request_id]
        return await asyncio.wait_for(future, timeout=self.timeout)

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> int:
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        self._pending[request_id] = loop.create_future()
        await self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        )
        return request_id

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise ACPError("ACP process is not running")
        self._debug_log("send", payload)
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        self._process.stdin.write(data.encode("utf-8") + b"\n")
        await self._process.stdin.drain()

    async def _read_stdout(self) -> None:
        process = self._process
        assert process is not None
        stdout = process.stdout
        assert stdout is not None
        while not self._closed:
            line = await stdout.readline()
            if not line:
                break
            try:
                message = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                self._debug_log("recv_invalid_json", {"line": line.decode("utf-8", "replace")})
                continue
            self._debug_log("recv", message)
            await self._handle_message(message)

        if not self._closed:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ACPError("ACP process exited"))

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = message["id"]
            future = self._pending.pop(request_id, None)
            if not future:
                return
            if "error" in message:
                future.set_exception(ACPError(str(message["error"])))
            else:
                future.set_result(message.get("result"))
            return

        if "id" in message and "method" in message:
            await self._handle_agent_request(message)
            return

        if message.get("method") == "session/update":
            params = message.get("params") or {}
            update = params.get("update")
            if isinstance(update, dict):
                if update.get("sessionUpdate") == "config_option_update":
                    self._set_config_options(update.get("configOptions"))
                await self._updates.put(update)

    async def _handle_agent_request(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        if method == "session/request_permission":
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "outcome": {
                            "outcome": "selected",
                            "optionId": self._select_permission_option(message),
                        }
                    },
                }
            )
            return

        await self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Unsupported ACP client method: {method}",
                },
            }
        )

    def _select_permission_option(self, message: dict[str, Any]) -> str:
        params = message.get("params") or {}
        options = params.get("options") or []
        wanted = self.permission

        for option in options:
            if option.get("optionId") == wanted or option.get("kind") == wanted:
                return str(option["optionId"])

        for kind in ("allow_once", "allow_always", "reject_once"):
            for option in options:
                if option.get("kind") == kind:
                    return str(option["optionId"])

        if options:
            return str(options[0].get("optionId"))
        return "allow-once"

    async def _drain_stderr(self) -> None:
        process = self._process
        assert process is not None
        stderr = process.stderr
        assert stderr is not None
        while not self._closed:
            line = await stderr.readline()
            if not line:
                break
            self._debug_log("stderr", {"line": line.decode("utf-8", "replace").rstrip("\n")})

    def _debug_log(self, direction: str, payload: Any) -> None:
        if not self._debug_log_path:
            return

        record = {
            "time": time.time(),
            "direction": direction,
            "payload": payload,
        }
        with contextlib.suppress(OSError, TypeError, ValueError):
            path = Path(self._debug_log_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    async def cancel(self) -> None:
        if self._session_id:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/cancel",
                    "params": {"sessionId": self._session_id},
                }
            )
