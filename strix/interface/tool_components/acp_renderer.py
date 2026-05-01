import shlex
from typing import Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from .base_renderer import BaseToolRenderer
from .registry import register_tool_renderer


MAX_OUTPUT_CHARS = 3000


@register_tool_renderer
class ACPRenderer(BaseToolRenderer):
    tool_name: ClassVar[str] = "acp"
    css_classes: ClassVar[list[str]] = ["tool-call", "acp-tool"]

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> Static:
        args = tool_data.get("args", {})
        result = tool_data.get("result")
        status = tool_data.get("status", "unknown")

        text = Text()
        icon, color = cls.status_icon(status)
        title = str(args.get("title") or "ACP tool")

        text.append("ACP ", style="dim")
        text.append(title, style="bold #60a5fa")
        text.append(" ")
        text.append(icon, style=color)

        cwd = cls._extract_cwd(args, result)
        if cwd:
            text.append("\n  cwd: ", style="dim")
            text.append(cwd)

        command = cls._extract_command(args, result)
        if command:
            text.append("\n  $ ", style="#22c55e")
            text.append(command)

        if isinstance(result, dict):
            exit_code = result.get("exit_code")
            if exit_code is not None:
                text.append("\n  exit: ", style="dim")
                text.append(str(exit_code))

            output = cls._extract_output(result)
            if output:
                text.append("\n")
                text.append(cls._truncate(output), style="dim")
        elif isinstance(result, str) and result.strip():
            text.append("\n")
            text.append(cls._truncate(result), style="dim")

        return Static(text, classes=cls.get_css_classes(status))

    @classmethod
    def _extract_cwd(cls, args: dict[str, Any], result: Any) -> str:
        if isinstance(result, dict) and result.get("cwd"):
            return str(result["cwd"])
        raw_input = args.get("input")
        if isinstance(raw_input, dict) and raw_input.get("cwd"):
            return str(raw_input["cwd"])
        return ""

    @classmethod
    def _extract_command(cls, args: dict[str, Any], result: Any) -> str:
        command: Any = None
        if isinstance(result, dict):
            command = result.get("command")
        if command is None:
            raw_input = args.get("input")
            if isinstance(raw_input, dict):
                command = raw_input.get("command")

        if isinstance(command, list):
            return shlex.join(str(part) for part in command)
        if isinstance(command, str):
            return command
        return ""

    @classmethod
    def _extract_output(cls, result: dict[str, Any]) -> str:
        output = result.get("formatted_output") or result.get("aggregated_output")
        if output:
            return str(output).strip()

        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        if stdout and stderr:
            return f"{stdout}\n{stderr}"
        return stdout or stderr

    @classmethod
    def _truncate(cls, output: str) -> str:
        if len(output) <= MAX_OUTPUT_CHARS:
            return output
        return output[:MAX_OUTPUT_CHARS].rstrip() + "\n... [truncated]"
