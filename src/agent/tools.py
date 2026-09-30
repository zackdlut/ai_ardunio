from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from arduino.compile_manager import compile_project
from arduino.upload_manager import upload_firmware
from config.settings import get_settings
from project.state_machine import ProjectStateMachine
from vision.component_db import normalize_class_name

_COMPONENT_ALIASES = {
    "电阻": "Resistor",
    "面包板": "Breadboard",
    "杜邦线": "JumperWire",
    "跳线": "JumperWire",
    "按键": "Button",
    "按钮": "Button",
    "开发板": "ArduinoUNO",
    "arduino": "ArduinoUNO",
    "uno": "ArduinoUNO",
}


@dataclass
class AgentContext:
    state_machine: ProjectStateMachine
    get_scene: Callable[[], dict[str, Any]]
    get_detections_json: Callable[[], list[dict]]
    selected_port: str = ""
    last_build_log: str = ""
    overlay_hint: dict[str, Any] = field(default_factory=dict)


def build_openai_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_scene_summary",
                "description": "当前摄像头检测与 scene graph 摘要",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_project_progress",
                "description": "当前实验项目步骤进度",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_next_instruction",
                "description": "下一步指导与叠加层提示",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compile_and_upload",
                "description": "编译并烧录当前步骤草图",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sketch_name": {
                            "type": "string",
                            "description": "firmware 子目录名，如 blink",
                        },
                        "port": {"type": "string", "description": "串口，如 COM3"},
                    },
                    "required": ["sketch_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "highlight_component",
                "description": "在摄像头画面上高亮器件或引脚，并显示一句短提示",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hint": {
                            "type": "string",
                            "description": "叠在画面上的短中文，12 字以内",
                        },
                        "component": {
                            "type": "string",
                            "description": "LED、Resistor、JumperWire、ArduinoUNO、Breadboard、Button 之一，可空",
                        },
                        "pin": {
                            "type": "string",
                            "description": "可选引脚，如 D13、GND、5V",
                        },
                    },
                    "required": ["hint"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_last_error",
                "description": "解释最近一次编译/烧录日志",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]


def build_tool_handlers(ctx: AgentContext) -> dict[str, Callable[..., str]]:
    settings = get_settings()

    def get_scene_summary() -> str:
        return json.dumps(ctx.get_scene(), ensure_ascii=False)

    def get_project_progress() -> str:
        return json.dumps(ctx.state_machine.progress(), ensure_ascii=False)

    def get_next_instruction() -> str:
        step = ctx.state_machine.next_step() or ctx.state_machine.current_step
        ctx.overlay_hint = {
            "hint": step.hint,
            "requires": step.requires,
            "pin": step.pin,
            "step_label": "下一步",
        }
        return json.dumps(
            {
                "next_step_id": step.id,
                "hint": step.hint,
                "requires": step.requires,
                "pin": step.pin,
            },
            ensure_ascii=False,
        )

    def compile_and_upload(sketch_name: str, port: str = "") -> str:
        use_port = port or ctx.selected_port
        if not use_port:
            return "未选择串口，请在界面选择开发板端口后重试"
        sketch_path = settings.firmware_dir / sketch_name
        if not sketch_path.is_dir():
            return f"草图目录不存在: {sketch_path}"

        compile_res = compile_project(sketch_path)
        log = compile_res.stdout + "\n" + compile_res.stderr
        ctx.last_build_log = log
        if not compile_res.success:
            return f"编译失败:\n{log[-4000:]}"

        upload_res = upload_firmware(sketch_path, use_port)
        log = upload_res.stdout + "\n" + upload_res.stderr
        ctx.last_build_log = log
        if not upload_res.success:
            return f"烧录失败:\n{log[-4000:]}"
        return "编译与烧录成功"

    def highlight_component(hint: str, component: str = "", pin: str = "") -> str:
        progress = ctx.state_machine.progress()
        component = (component or "").strip()
        pin_name = (pin or "").strip()
        if component:
            component = _COMPONENT_ALIASES.get(component.lower(), component)
            component = normalize_class_name(component)
        requires = [component] if component else []
        ctx.overlay_hint = {
            "hint": (hint or "").strip(),
            "requires": requires,
            "pin": pin_name or None,
            "step_label": f"第 {progress['step_index'] + 1}/{progress['total_steps']} 步",
        }
        return "已在画面标出：" + (hint or "").strip()

    def explain_last_error() -> str:
        if not ctx.last_build_log:
            return "暂无编译/烧录日志"
        return ctx.last_build_log[-6000:]

    return {
        "get_scene_summary": lambda **_: get_scene_summary(),
        "get_project_progress": lambda **_: get_project_progress(),
        "get_next_instruction": lambda **_: get_next_instruction(),
        "compile_and_upload": lambda **kw: compile_and_upload(**kw),
        "highlight_component": lambda **kw: highlight_component(**kw),
        "explain_last_error": lambda **_: explain_last_error(),
    }
