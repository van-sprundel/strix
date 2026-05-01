import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from strix.interface.utils import configure_acp_cwd
from strix.llm.acp import ACPClient, ACPError
from strix.llm.config import LLMConfig
from strix.llm.llm import LLM
from strix.llm.mcp_server import _call_tool, _list_tools


FAKE_ACP = r"""
import json
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"authMethods": [{"id": "chatgpt", "name": "ChatGPT"}]},
        })
    elif method == "authenticate":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/new":
        assert request["params"]["mcpServers"]
        assert request["params"]["mcpServers"][0]["name"] == "strix"
        assert request["params"]["mcpServers"][0]["args"][0].endswith("mcp_server.py")
        assert {
            item["name"]: item["value"] for item in request["params"]["mcpServers"][0]["env"]
        }["STRIX_MCP_DEBUG_LOG"].endswith("acp-debug.mcp.jsonl")
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "sessionId": "session-1",
                "configOptions": [
                    {
                        "id": "model",
                        "name": "Model",
                        "category": "model",
                        "type": "select",
                        "currentValue": "gpt-5.4",
                        "options": [
                            {"value": "gpt-5.4", "name": "GPT-5.4"},
                            {"value": "gpt-5.5", "name": "GPT-5.5"},
                        ],
                    },
                    {
                        "id": "reasoning-effort",
                        "name": "Reasoning",
                        "category": "thought_level",
                        "type": "select",
                        "currentValue": "medium",
                        "options": [
                            {"value": "medium", "name": "Medium"},
                            {"value": "high", "name": "High"},
                        ],
                    },
                ],
            },
        })
    elif method == "session/set_config_option":
        assert request["params"]["sessionId"] == "session-1"
        assert (
            request["params"] in [
                {"sessionId": "session-1", "configId": "model", "value": "gpt-5.5"},
                {
                    "sessionId": "session-1",
                    "configId": "reasoning-effort",
                    "value": "high",
                },
            ]
        )
        send({"jsonrpc": "2.0", "id": request_id, "result": {"configOptions": []}})
    elif method == "session/prompt":
        prompt = request["params"]["prompt"][0]["text"]
        assert "Keep shell output bounded" in prompt
        assert "mcp__strix__terminal_execute" in prompt
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "# Executive Summary\nOK\n"},
                },
            },
        })
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "git status --short",
                    "kind": "terminal",
                    "rawInput": {"command": "git status --short", "cwd": "/tmp/project"},
                },
            },
        })
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-1",
                    "status": "completed",
                    "rawOutput": json.dumps({
                        "command": ["git", "status", "--short"],
                        "cwd": "/tmp/project",
                        "stdout": "",
                        "stderr": "",
                        "exit_code": 0,
                    }),
                },
            },
        })
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "\n# Methodology\nChecked status.\n"
                                "# Technical Analysis\nNo issues.\n"
                                "# Recommendations\nNone.\n",
                    },
                },
            },
        })
        send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})
    elif method == "session/close":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
"""


FAKE_ACP_HANG = r"""
import json
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "session-1"}})
    elif method == "session/prompt":
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "rg stuck",
                    "kind": "search",
                },
            },
        })
    elif method == "session/close":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
"""


FAKE_ACP_SILENT_THEN_DONE = r"""
import json
import sys
import time


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "session-1"}})
    elif method == "session/prompt":
        send({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "session-1",
                "update": {
                    "sessionUpdate": "available_commands_update",
                    "availableCommands": [],
                },
            },
        })
        time.sleep(0.2)
        send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})
    elif method == "session/close":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
"""


FAKE_ACP_TERMINAL = r"""
import json
import os
import sys


prompt_request_id = None
terminal_id = None


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    request_id = request.get("id")

    if method == "initialize":
        capabilities = request["params"]["clientCapabilities"]
        assert capabilities["terminal"] is True
        assert capabilities["fs"]["readTextFile"] is True
        assert capabilities["fs"]["writeTextFile"] is False
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
    elif method == "session/new":
        send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "session-1"}})
    elif method == "session/prompt":
        prompt_request_id = request_id
        send({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "terminal/create",
            "params": {
                "sessionId": "session-1",
                "command": sys.executable,
                "args": ["-c", "print('owned by strix')"],
                "cwd": os.getcwd(),
                "outputByteLimit": 4096,
            },
        })
    elif request_id == 100:
        terminal_id = request["result"]["terminalId"]
        assert terminal_id
        send({
            "jsonrpc": "2.0",
            "id": 101,
            "method": "terminal/wait_for_exit",
            "params": {"sessionId": "session-1", "terminalId": terminal_id},
        })
    elif request_id == 101:
        assert request["result"]["exitCode"] == 0
        send({
            "jsonrpc": "2.0",
            "id": 102,
            "method": "terminal/output",
            "params": {"sessionId": "session-1", "terminalId": terminal_id},
        })
    elif request_id == 102:
        assert "owned by strix" in request["result"]["output"]
        assert request["result"]["truncated"] is False
        assert request["result"]["exitStatus"]["exitCode"] == 0
        send({
            "jsonrpc": "2.0",
            "id": 103,
            "method": "fs/read_text_file",
            "params": {
                "sessionId": "session-1",
                "path": os.path.join(os.getcwd(), "fixture.txt"),
                "line": 2,
                "limit": 1,
            },
        })
    elif request_id == 103:
        assert request["result"]["content"] == "second\n"
        send({
            "jsonrpc": "2.0",
            "id": 104,
            "method": "terminal/release",
            "params": {"sessionId": "session-1", "terminalId": terminal_id},
        })
    elif request_id == 104:
        send({"jsonrpc": "2.0", "id": prompt_request_id, "result": {"stopReason": "end_turn"}})
    elif method == "session/close":
        send({"jsonrpc": "2.0", "id": request_id, "result": {}})
"""


@pytest.mark.asyncio
async def test_acp_backend_streams_fake_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_acp = tmp_path / "fake_acp.py"
    debug_log = tmp_path / "acp-debug.jsonl"
    fake_acp.write_text(FAKE_ACP, encoding="utf-8")

    monkeypatch.setenv("STRIX_LLM", "acp/codex")
    monkeypatch.setenv("STRIX_ACP_COMMAND", f"{sys.executable} {fake_acp}")
    monkeypatch.setenv("STRIX_ACP_CWD", str(tmp_path))
    monkeypatch.setenv("STRIX_ACP_MODEL", "gpt-5.5")
    monkeypatch.setenv("STRIX_ACP_REASONING_EFFORT", "high")
    monkeypatch.setenv("STRIX_ACP_DEBUG_LOG", str(debug_log))

    llm = LLM(LLMConfig(), agent_name=None)
    responses = [
        response
        async for response in llm.generate([{"role": "user", "content": "Assess this repo"}])
    ]

    assert responses
    assert responses[-1].stop_requested is True
    assert "Executive Summary" in responses[-1].content
    assert "Recommendations" in responses[-1].content
    assert llm._total_stats.requests == 1
    assert '"direction": "send"' in debug_log.read_text(encoding="utf-8")
    assert '"method": "session/update"' in debug_log.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_acp_prompt_idle_timeout_cancels_stuck_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_acp = tmp_path / "fake_acp_hang.py"
    fake_acp.write_text(FAKE_ACP_HANG, encoding="utf-8")

    monkeypatch.setenv("STRIX_ACP_COMMAND", f"{sys.executable} {fake_acp}")
    monkeypatch.setenv("STRIX_ACP_CWD", str(tmp_path))
    monkeypatch.setenv("STRIX_ACP_IDLE_TIMEOUT", "0.1")

    client = ACPClient(agent="codex", timeout=5)
    try:
        with pytest.raises(ACPError, match="Last active tool: rg stuck"):
            async for _ in client.prompt("hang"):
                pass
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_acp_prompt_idle_timeout_allows_model_silence_without_active_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_acp = tmp_path / "fake_acp_silent.py"
    fake_acp.write_text(FAKE_ACP_SILENT_THEN_DONE, encoding="utf-8")

    monkeypatch.setenv("STRIX_ACP_COMMAND", f"{sys.executable} {fake_acp}")
    monkeypatch.setenv("STRIX_ACP_CWD", str(tmp_path))
    monkeypatch.setenv("STRIX_ACP_IDLE_TIMEOUT", "0.1")

    client = ACPClient(agent="codex", timeout=5)
    try:
        updates = [update async for update in client.prompt("think")]
    finally:
        await client.close()

    assert updates[-1]["sessionUpdate"] == "turn_complete"


@pytest.mark.asyncio
async def test_acp_client_handles_terminal_and_read_file_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_acp = tmp_path / "fake_acp_terminal.py"
    fake_acp.write_text(FAKE_ACP_TERMINAL, encoding="utf-8")
    (tmp_path / "fixture.txt").write_text("first\nsecond\nthird\n", encoding="utf-8")

    monkeypatch.setenv("STRIX_ACP_COMMAND", f"{sys.executable} {fake_acp}")
    monkeypatch.setenv("STRIX_ACP_CWD", str(tmp_path))
    monkeypatch.setenv("STRIX_ACP_IDLE_TIMEOUT", "0.1")

    client = ACPClient(agent="codex", timeout=5)
    try:
        updates = [update async for update in client.prompt("run a command")]
    finally:
        await client.close()

    assert updates[-1]["sessionUpdate"] == "turn_complete"


def test_configure_acp_cwd_uses_first_local_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "repo"
    target.mkdir()

    monkeypatch.setenv("STRIX_LLM", "acp/codex")
    monkeypatch.delenv("STRIX_ACP_CWD", raising=False)

    configure_acp_cwd([{"source_path": str(target), "workspace_subdir": "repo"}])

    assert Path(str(os.getenv("STRIX_ACP_CWD"))).resolve() == target.resolve()


@pytest.mark.asyncio
async def test_strix_mcp_server_exposes_local_tools() -> None:
    listed = _list_tools()
    names = {tool["name"] for tool in listed}

    assert "think" in names
    assert "terminal_execute" in names

    result = await _call_tool("think", {"thought": "exercise the MCP bridge"})

    assert result["isError"] is False
    assert "Thought recorded" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_strix_mcp_terminal_execute_runs_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIX_MCP_CWD", str(tmp_path))
    (tmp_path / "fixture.txt").write_text("owned by mcp\n", encoding="utf-8")

    result = await _call_tool(
        "terminal_execute",
        {"command": "sed -n '1p' fixture.txt", "timeout": 5},
    )

    assert result["isError"] is False
    assert "owned by mcp" in result["content"][0]["text"]


def test_strix_mcp_server_responds_over_stdio() -> None:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "probe", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    wire = b""
    for message in messages:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        wire += b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
        wire += payload

    process = subprocess.run(  # noqa: S603
        [sys.executable, str(Path("strix/llm/mcp_server.py").resolve())],
        input=wire,
        capture_output=True,
        timeout=10,
        check=True,
    )

    assert b'"name":"terminal_execute"' in process.stdout


def test_strix_mcp_server_writes_debug_log(tmp_path: Path) -> None:
    debug_log = tmp_path / "mcp-debug.jsonl"
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    wire = b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n" + payload
    env = os.environ.copy()
    env["STRIX_MCP_DEBUG_LOG"] = str(debug_log)

    subprocess.run(  # noqa: S603
        [sys.executable, str(Path("strix/llm/mcp_server.py").resolve())],
        input=wire,
        capture_output=True,
        timeout=10,
        check=True,
        env=env,
    )

    written = debug_log.read_text(encoding="utf-8")
    assert '"direction": "startup"' in written
    assert '"method": "tools/list"' in written
