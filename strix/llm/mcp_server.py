import argparse
import asyncio
import inspect
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, get_args, get_origin

import defusedxml.ElementTree as DefusedET

from strix.tools import execute_tool_with_validation
from strix.tools.registry import get_tool_param_schema, needs_agent_state, tools


PROTOCOL_VERSION = "2024-11-05"
MCP_SESSION_ID = "strix"
EXCLUDED_TOOLS = {
    "create_agent",
    "send_message_to_agent",
    "wait_for_message",
    "agent_finish",
    "finish_scan",
    "view_agent_graph",
    "stop_agent",
    "send_user_message_to_agent",
}
TERMINAL_OUTPUT_LIMIT = 64_000
TERMINAL_TIMEOUT = 60.0


def _debug_log(direction: str, payload: Any) -> None:
    log_path = os.getenv("STRIX_MCP_DEBUG_LOG")
    if not log_path:
        return
    entry = {
        "time": time.time(),
        "pid": os.getpid(),
        "direction": direction,
        "payload": payload,
    }
    try:
        with Path(log_path).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break
        decoded = line.decode("ascii", errors="replace")
        key, _, value = decoded.partition(":")
        headers[key.lower()] = value.strip()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None
    payload = sys.stdin.buffer.read(content_length)
    return json.loads(payload.decode("utf-8"))


def _write_message(message: dict[str, Any]) -> None:
    _debug_log("send", message)
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _json_response_payload(message: dict[str, Any]) -> bytes:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _response(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _is_exposed_tool(tool: dict[str, Any]) -> bool:
    name = str(tool.get("name") or "")
    if not name or name in EXCLUDED_TOOLS:
        return False
    if bool(tool.get("sandbox_execution", True)):
        return False
    return not needs_agent_state(name)


def _tool_description(tool: dict[str, Any]) -> str:
    xml_schema = tool.get("xml_schema")
    if not isinstance(xml_schema, str):
        return "Strix tool"
    try:
        root = DefusedET.fromstring(xml_schema)
    except DefusedET.ParseError:
        return "Strix tool"

    parts = []
    for tag_name in ("description", "details"):
        node = root.find(tag_name)
        if node is not None and node.text and node.text.strip():
            parts.append(node.text.strip())
    return "\n\n".join(parts) or "Strix tool"


def _json_type(annotation: Any) -> str:  # noqa: PLR0911
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is inspect.Signature.empty:
        return "string"
    if annotation is bool:
        return "boolean"
    if annotation in {int, float}:
        return "number" if annotation is float else "integer"
    if annotation in {dict, list}:
        return "object" if annotation is dict else "array"
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if origin in {type(None), None}:
        return "null"
    if args:
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
    return "string"


def _input_schema(tool: dict[str, Any]) -> dict[str, Any]:
    func = tool.get("function")
    if not callable(func):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    sig = inspect.signature(func)
    param_schema = get_tool_param_schema(str(tool.get("name"))) or {}
    required_names = set(param_schema.get("required", set()))

    properties: dict[str, Any] = {}
    required = []
    for name, param in sig.parameters.items():
        if name == "agent_state":
            continue
        properties[name] = {"type": _json_type(param.annotation)}
        if name in required_names or param.default is inspect.Signature.empty:
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _list_tools() -> list[dict[str, Any]]:
    listed = [
        {
            "name": "terminal_execute",
            "description": (
                "Execute a bounded shell command in the target repository. Use this for "
                "all repository searches, file reads, and CLI commands when Strix runs "
                "through ACP. Output is capped and long-running commands time out."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run with bash -lc.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Optional working directory. Defaults to STRIX_MCP_CWD or "
                            "the MCP server current directory."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Optional timeout in seconds, capped at 60.",
                    },
                    "max_output_chars": {
                        "type": "integer",
                        "description": "Optional output cap, capped at 64000 characters.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    ]
    listed.extend(
        {
            "name": str(tool["name"]),
            "description": _tool_description(tool),
            "inputSchema": _input_schema(tool),
        }
        for tool in tools
        if _is_exposed_tool(tool)
    )
    return listed


async def _execute_terminal(arguments: dict[str, Any]) -> dict[str, Any]:
    command = str(arguments.get("command") or "")
    _debug_log("terminal_execute", {"command": command, "cwd": arguments.get("cwd")})
    if not command.strip():
        return {
            "content": [{"type": "text", "text": "Command must not be empty"}],
            "isError": True,
        }

    cwd = str(arguments.get("cwd") or os.getenv("STRIX_MCP_CWD") or Path.cwd())
    timeout = min(float(arguments.get("timeout") or TERMINAL_TIMEOUT), TERMINAL_TIMEOUT)
    output_limit = min(
        int(arguments.get("max_output_chars") or TERMINAL_OUTPUT_LIMIT),
        TERMINAL_OUTPUT_LIMIT,
    )

    process = await asyncio.create_subprocess_exec(
        "/usr/bin/bash",
        "-lc",
        command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        timed_out = False
    except TimeoutError:
        process.kill()
        stdout, stderr = await process.communicate()
        timed_out = True

    stdout_text = stdout.decode("utf-8", "replace")
    stderr_text = stderr.decode("utf-8", "replace")
    combined = stdout_text
    if stderr_text:
        combined = (
            f"{combined}\n[stderr]\n{stderr_text}" if combined else f"[stderr]\n{stderr_text}"
        )

    truncated = len(combined) > output_limit
    if truncated:
        combined = combined[-output_limit:]
        combined = f"[output truncated to last {output_limit} chars]\n{combined}"

    result = {
        "command": command,
        "argv": ["/usr/bin/bash", "-lc", command],
        "cwd": cwd,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "truncated": truncated,
        "output": combined,
    }
    return {
        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
        "isError": timed_out or process.returncode not in {0, None},
    }


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "terminal_execute":
        return await _execute_terminal(arguments)

    if name not in {tool["name"] for tool in tools if _is_exposed_tool(tool)}:
        return {
            "content": [{"type": "text", "text": f"Tool '{name}' is not available"}],
            "isError": True,
        }

    result = await execute_tool_with_validation(name, None, **arguments)
    is_error = isinstance(result, str) and result.lower().startswith("error:")
    if isinstance(result, dict) and result.get("success") is False:
        is_error = True

    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(result, ensure_ascii=False, indent=2)
                if not isinstance(result, str)
                else result,
            }
        ],
        "isError": is_error,
    }


async def _handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    if request_id is None:
        return None

    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "strix", "version": "0.8.3"},
            },
        )
    if method == "tools/list":
        return _response(request_id, {"tools": _list_tools()})
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "Tool arguments must be an object")
        return _response(request_id, await _call_tool(name, arguments))

    return _error(request_id, -32601, f"Unsupported MCP method: {method}")


def _run() -> None:
    _debug_log(
        "startup",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "target_cwd": os.getenv("STRIX_MCP_CWD"),
        },
    )
    while True:
        try:
            message = _read_message()
            if message is None:
                _debug_log("shutdown", {"reason": "stdin_closed"})
                return
            _debug_log("recv", message)
            response = asyncio.run(_handle_request(message))
            if response is not None:
                _write_message(response)
        except Exception as e:
            _debug_log("error", {"type": type(e).__name__, "message": str(e)})
            raise


class MCPHTTPHandler(BaseHTTPRequestHandler):
    server_version = "StrixMCP/0.1"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def do_DELETE(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "MCP endpoint accepts POST requests")

    def do_POST(self) -> None:
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown MCP endpoint")
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length)
            message = json.loads(body.decode("utf-8"))
            _debug_log(
                "http_recv",
                {
                    "headers": dict(self.headers.items()),
                    "body": message,
                },
            )
            response = asyncio.run(_handle_request(message))
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            _debug_log("http_error", {"type": type(e).__name__, "message": str(e)})
            response = _error(None, -32700, str(e))

        if response is None:
            self.send_response(HTTPStatus.ACCEPTED)
            self._send_common_headers()
            self.end_headers()
            return

        _debug_log("http_send", response)
        payload = _json_response_payload(response)
        self.send_response(HTTPStatus.OK)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, DELETE")
        self.send_header("Mcp-Session-Id", MCP_SESSION_ID)

    def log_message(self, message_format: str, *args: Any) -> None:
        _debug_log("http_access", {"message": message_format % args})


def _run_http(host: str, port: int) -> None:
    _debug_log(
        "http_startup",
        {
            "host": host,
            "port": port,
            "cwd": str(Path.cwd()),
            "target_cwd": os.getenv("STRIX_MCP_CWD"),
        },
    )
    ThreadingHTTPServer((host, port), MCPHTTPHandler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", nargs=2, metavar=("HOST", "PORT"))
    args = parser.parse_args()
    if args.http:
        _run_http(args.http[0], int(args.http[1]))
    else:
        _run()


if __name__ == "__main__":
    main()
