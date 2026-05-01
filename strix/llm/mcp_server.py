import asyncio
import inspect
import json
import sys
from typing import Any, get_args, get_origin

import defusedxml.ElementTree as DefusedET

from strix.tools import execute_tool_with_validation
from strix.tools.registry import get_tool_param_schema, needs_agent_state, tools


PROTOCOL_VERSION = "2024-11-05"
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
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


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
    return [
        {
            "name": str(tool["name"]),
            "description": _tool_description(tool),
            "inputSchema": _input_schema(tool),
        }
        for tool in tools
        if _is_exposed_tool(tool)
    ]


async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


async def _run() -> None:
    while True:
        message = await asyncio.to_thread(_read_message)
        if message is None:
            return
        response = await _handle_request(message)
        if response is not None:
            _write_message(response)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
