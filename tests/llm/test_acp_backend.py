import os
import sys
from pathlib import Path

import pytest

from strix.interface.utils import configure_acp_cwd
from strix.llm.config import LLMConfig
from strix.llm.llm import LLM


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
        send({"jsonrpc": "2.0", "id": request_id, "result": {"sessionId": "session-1"}})
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
