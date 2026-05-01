import os
import sys
from pathlib import Path

import pytest

from strix.interface.utils import configure_acp_cwd
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


@pytest.mark.asyncio
async def test_acp_backend_streams_fake_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_acp = tmp_path / "fake_acp.py"
    fake_acp.write_text(FAKE_ACP, encoding="utf-8")

    monkeypatch.setenv("STRIX_LLM", "acp/codex")
    monkeypatch.setenv("STRIX_ACP_COMMAND", f"{sys.executable} {fake_acp}")
    monkeypatch.setenv("STRIX_ACP_CWD", str(tmp_path))
    monkeypatch.setenv("STRIX_ACP_MODEL", "gpt-5.5")
    monkeypatch.setenv("STRIX_ACP_REASONING_EFFORT", "high")

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
    assert "terminal_execute" not in names

    result = await _call_tool("think", {"thought": "exercise the MCP bridge"})

    assert result["isError"] is False
    assert "Thought recorded" in result["content"][0]["text"]
