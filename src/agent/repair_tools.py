from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from repair.annotations import (
    MAX_ANNOTATIONS,
    make_shape,
    make_shapes,
    merge_annotations,
    normalize_box,
    shape_summary,
)
from repair.case_manager import CaseManager
from repair.debug_log import RepairDebugLog
from repair.documents import DocumentLibrary
from repair.geometry import warp_points, warp_shapes
from repair.ifixit import IFixitClient

_STATUS_KIND = {"fail": "fault", "pass": "ok", "skip": "ok"}
MAX_STEP_SHAPES = 6


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _warp_box(box: list[float] | None, H: np.ndarray | None) -> list[float] | None:
    if not box or H is None:
        return box
    x1, y1, x2, y2 = box
    pts = warp_points(H, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    xs = [min(max(p[0], 0.0), 1.0) for p in pts]
    ys = [min(max(p[1], 0.0), 1.0) for p in pts]
    return [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]


def _grow_box(box: list[float], pad: float) -> list[float]:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [
        max(0.0, x1 - w * pad),
        max(0.0, y1 - h * pad),
        min(1.0, x2 + w * pad),
        min(1.0, y2 + h * pad),
    ]


def _from_zoom(shape: dict[str, Any], region: list[float]) -> dict[str, Any]:
    rx1, ry1, rx2, ry2 = region
    out = dict(shape)
    out["pts"] = [
        [round(rx1 + float(x) * (rx2 - rx1), 5), round(ry1 + float(y) * (ry2 - ry1), 5)]
        for x, y in shape.get("pts") or []
    ]
    return out


def _guide_summary(guide: dict[str, Any] | None) -> dict[str, Any] | None:
    if not guide:
        return None
    return {
        "id": guide["id"],
        "title": guide["title"],
        "status": guide["status"],
        "current_step": int(guide.get("current_step") or 0) + 1,
        "steps": [
            {
                "step": i + 1,
                "title": s["title"],
                "status": s["status"],
                "verify": s["verify"],
                "feedback": s["feedback"],
                "shapes": [shape_summary(x) for x in s.get("shapes") or []],
            }
            for i, s in enumerate(guide.get("steps") or [])
        ],
    }


@dataclass
class RepairContext:
    """State shared by the repair tools (LLM worker thread) and the repair panel (GUI thread)."""

    cases: CaseManager
    library: DocumentLibrary
    ifixit: IFixitClient | None = None
    box_format: str = "norm1000"
    doc_max_side: int = 1600
    case_id: int | None = None
    image_size: tuple[int, int] | None = None
    on_change: Callable[[str], None] | None = None
    _annotations: list[dict[str, Any]] = field(default_factory=list)
    _board_box: list[float] | None = None
    _guide: dict[str, Any] | None = None
    _pending_images: list[tuple[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.RLock)
    # Full-resolution reference frame of the current turn; zoom_view crops it.
    anchor_frame: np.ndarray | None = None
    zoom_max_side: int = 1280
    debug: RepairDebugLog | None = None
    _zooms: dict[int, dict[str, Any]] = field(default_factory=dict)

    def begin_turn(self, anchor_frame: np.ndarray | None, image_size: tuple[int, int] | None) -> None:
        with self._lock:
            self.anchor_frame = anchor_frame
            self.image_size = image_size
            self._zooms = {}

    # -- zoom views -----------------------------------------------------------

    def zoom(self, box: list[float]) -> tuple[int, str, str]:
        """Crop ``box`` (0..1 of the attached image) from the anchor frame.

        Returns ``(zoom_id, caption, data_url)``; the crop is enlarged so small
        parts get more of the vision encoder's attention.
        """
        frame = self.anchor_frame
        if frame is None:
            raise ValueError("这一轮没有附画面，无法放大")
        x1, y1, x2, y2 = _grow_box(box, 0.06)
        h, w = frame.shape[:2]
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = max(px1 + 8, int(round(x2 * w))), max(py1 + 8, int(round(y2 * h)))
        crop = frame[py1:py2, px1:px2]
        ch, cw = crop.shape[:2]
        scale = min(4.0, self.zoom_max_side / max(cw, ch))
        if abs(scale - 1.0) > 0.05:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=interp)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("放大图编码失败")
        region = [px1 / w, py1 / h, min(1.0, px2 / w), min(1.0, py2 / h)]
        with self._lock:
            zoom_id = len(self._zooms) + 1
            self._zooms[zoom_id] = {"region": region, "size": (crop.shape[1], crop.shape[0])}
        url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        caption = f"放大视图 zoom_id={zoom_id}（原图 {crop.shape[1]}x{crop.shape[0]} 像素的局部，已放大）"
        return zoom_id, caption, url

    def zoom_info(self, zoom_id: Any) -> dict[str, Any] | None:
        if zoom_id in (None, "", 0):
            return None
        with self._lock:
            info = self._zooms.get(int(zoom_id))
        if info is None:
            raise ValueError(f"zoom_id={zoom_id} 不存在，先调用 zoom_view，且只在本轮有效")
        return info

    def make_shapes(self, items: Any, zoom_id: Any = None) -> tuple[list[dict[str, Any]], int]:
        """Shapes from tool items; coordinates may be relative to a zoom view."""
        info = self.zoom_info(zoom_id)
        if info is None:
            return make_shapes(items or [], self.box_format, self.image_size)
        made, skipped = make_shapes(items or [], self.box_format, info["size"])
        return [_from_zoom(s, info["region"]) for s in made], skipped

    def normalize_box(self, box: Any, zoom_id: Any = None) -> list[float] | None:
        info = self.zoom_info(zoom_id)
        if info is None:
            return normalize_box(box, self.box_format, self.image_size)
        local = normalize_box(box, self.box_format, info["size"])
        if local is None:
            return None
        rx1, ry1, rx2, ry2 = info["region"]
        sx, sy = rx2 - rx1, ry2 - ry1
        return [
            round(rx1 + local[0] * sx, 4),
            round(ry1 + local[1] * sy, 4),
            round(rx1 + local[2] * sx, 4),
            round(ry1 + local[3] * sy, 4),
        ]

    # -- shared state -------------------------------------------------------

    @property
    def annotations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._annotations]

    def set_annotations(self, items: list[dict[str, Any]]) -> None:
        with self._lock:
            self._annotations = [dict(item) for item in items]
        self.notify("annotations")

    def add_annotations(self, items: list[dict[str, Any]], replace: bool = False) -> None:
        with self._lock:
            base = [] if replace else self._annotations
            self._annotations = merge_annotations(base, items)
        self.notify("annotations")

    def queue_image(self, caption: str, url: str) -> None:
        with self._lock:
            self._pending_images.append((caption, url))
        if self.debug is not None:
            self.debug.image(caption, url)

    def drain_images(self) -> list[tuple[str, str]]:
        with self._lock:
            images, self._pending_images = self._pending_images, []
        return images

    def case(self) -> dict[str, Any] | None:
        return self.cases.get_case(self.case_id) if self.case_id is not None else None

    def documents(self) -> list[dict[str, Any]]:
        return self.cases.list_documents(self.case_id)

    def ensure_case(self) -> int:
        if self.case_id is None or self.cases.get_case(self.case_id) is None:
            self.case_id = self.cases.create_case()["id"]
            self.notify("case")
        return self.case_id

    def new_case(self, device: str = "", symptom: str = "") -> dict[str, Any]:
        if self.case_id is not None:
            self.cases.update_case(self.case_id, status="closed")
        case = self.cases.create_case(device, symptom)
        self.case_id = case["id"]
        with self._lock:
            self._annotations = []
            self._board_box = None
            self._guide = None
        self.notify("case")
        self.notify("guide")
        self.notify("annotations")
        return case

    # -- AR shapes and the reference frame ------------------------------------

    @property
    def board_box(self) -> list[float] | None:
        with self._lock:
            if self._guide and self._guide.get("board_box"):
                return list(self._guide["board_box"])
            return list(self._board_box) if self._board_box else None

    def set_board_box(self, box: list[float] | None) -> None:
        with self._lock:
            self._board_box = list(box) if box else None

    def current_shapes(self) -> list[dict[str, Any]]:
        """Free annotations plus the shapes of the current guide step."""
        with self._lock:
            shapes = [dict(item) for item in self._annotations]
            step = self._current_step_locked()
            if step is not None:
                shapes += [dict(s) for s in step.get("shapes") or []]
            return shapes

    def rebase(self, H: np.ndarray | None) -> None:
        """The reference frame moved by ``H``: carry every stored shape along."""
        if H is None:
            return
        with self._lock:
            self._annotations = warp_shapes(self._annotations, H)
            self._board_box = _warp_box(self._board_box, H)
            guide = self._guide
            if guide is not None:
                for step in guide["steps"]:
                    step["shapes"] = warp_shapes(step.get("shapes") or [], H)
                guide["board_box"] = _warp_box(guide.get("board_box"), H)
                steps = [dict(s) for s in guide["steps"]]
                guide_id = guide["id"]
            else:
                guide_id = None
        if guide_id is not None:
            self.cases.save_steps(guide_id, steps)

    # -- step-by-step guide ---------------------------------------------------

    @property
    def guide(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self._guide)) if self._guide else None

    def load_guide(self) -> dict[str, Any] | None:
        guide = self.cases.active_guide(self.case_id) if self.case_id is not None else None
        with self._lock:
            self._guide = guide
        self.notify("guide")
        self.notify("annotations")
        return guide

    def current_step(self) -> tuple[int, dict[str, Any]] | None:
        with self._lock:
            step = self._current_step_locked()
            if step is None:
                return None
            return int(self._guide["current_step"]), dict(step)  # type: ignore[index]

    def _current_step_locked(self) -> dict[str, Any] | None:
        guide = self._guide
        if not guide or guide.get("status") != "active" or not guide.get("steps"):
            return None
        index = min(max(int(guide.get("current_step") or 0), 0), len(guide["steps"]) - 1)
        return guide["steps"][index]

    def start_guide(
        self,
        title: str,
        steps: list[dict[str, Any]],
        check_id: int | None = None,
        board_box: list[float] | None = None,
    ) -> dict[str, Any]:
        case_id = self.ensure_case()
        guide = self.cases.start_guide(case_id, title, steps, check_id=check_id, board_box=board_box)
        with self._lock:
            self._guide = guide
        self._guide_changed()
        return guide

    def goto_step(self, index: int) -> dict[str, Any] | None:
        guide_id = self._guide_id()
        if guide_id is None:
            return None
        guide = self.cases.set_step(guide_id, index)
        with self._lock:
            self._guide = guide
        self._guide_changed()
        return guide

    def advance(self, mark: str = "skipped") -> dict[str, Any] | None:
        """Leave the current step (marking it ``mark`` if still pending).

        Past the last step the guide is finished. Returns the guide, or None if
        there is no active guide.
        """
        current = self.current_step()
        guide_id = self._guide_id()
        if current is None or guide_id is None:
            return None
        index, step = current
        if step.get("status") in ("pending", "failed") and mark:
            self.cases.update_step(guide_id, index, status=mark)
        guide = self.cases.get_guide(guide_id)
        if guide and index + 1 < len(guide["steps"]):
            guide = self.cases.set_step(guide_id, index + 1)
        else:
            guide = self.cases.end_guide(guide_id, "done")
        with self._lock:
            self._guide = guide if guide and guide["status"] == "active" else None
        self._guide_changed()
        return guide

    def back(self) -> dict[str, Any] | None:
        current = self.current_step()
        if current is None:
            return None
        return self.goto_step(max(0, current[0] - 1))

    def update_step(
        self,
        index: int | None = None,
        shapes: list[dict[str, Any]] | None = None,
        replace_shapes: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        guide_id = self._guide_id()
        current = self.current_step()
        if guide_id is None or current is None:
            raise ValueError("当前没有进行中的指引")
        index = current[0] if index is None else int(index)
        guide = self.cases.get_guide(guide_id)
        if guide is None or not 0 <= index < len(guide["steps"]):
            raise ValueError(f"步骤不存在: 第 {index + 1} 步")
        if shapes is not None:
            old = [] if replace_shapes else guide["steps"][index]["shapes"]
            fields["shapes"] = merge_annotations(old, shapes, limit=MAX_STEP_SHAPES)
        guide = self.cases.update_step(guide_id, index, **fields)
        with self._lock:
            self._guide = guide
        self._guide_changed()
        return guide

    def confirm(
        self, passed: bool, feedback: str = "", shapes: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        current = self.current_step()
        if current is None:
            raise ValueError("当前没有进行中的指引")
        index, _step = current
        if passed:
            self.update_step(index, status="done", feedback=feedback)
            guide = self.advance(mark="")
            nxt = self.current_step()
            return {
                "step": index + 1,
                "passed": True,
                "finished": nxt is None,
                "next_step": None if nxt is None else {"step": nxt[0] + 1, "title": nxt[1]["title"]},
                "guide": guide,
            }
        self.update_step(
            index, shapes=shapes or None, replace_shapes=False, status="failed", feedback=feedback
        )
        return {"step": index + 1, "passed": False, "feedback": feedback}

    def end_guide(self, summary: str = "", status: str = "done") -> dict[str, Any] | None:
        guide_id = self._guide_id()
        if guide_id is None:
            return None
        guide = self.cases.end_guide(guide_id, status, summary)
        with self._lock:
            self._guide = None
        self._guide_changed()
        return guide

    def _guide_id(self) -> int | None:
        with self._lock:
            return int(self._guide["id"]) if self._guide else None

    def _guide_changed(self) -> None:
        self.notify("guide")
        self.notify("annotations")

    # -- operations used by both tools and the panel -------------------------

    def record_result(
        self, check_id: int, status: str, measured: str = "", note: str = ""
    ) -> dict[str, Any]:
        case_id = self.ensure_case()
        check = self.cases.record_result(case_id, int(check_id), status, measured, note)
        kind = _STATUS_KIND.get(status)
        if kind:
            with self._lock:
                for item in self._annotations:
                    if item.get("check_id") == check["id"]:
                        item["kind"] = kind
                        if measured.strip():
                            item["note"] = f"实测 {measured.strip()}"[:60]
        nxt = self.cases.next_pending(case_id, check["id"])
        self.cases.set_current(case_id, nxt["id"] if nxt else None)
        self.notify("case")
        self.notify("annotations")
        return {"recorded": check, "next_check": nxt}

    def focus(self, check_id: int | None) -> dict[str, Any] | None:
        case_id = self.ensure_case()
        check = self.cases.set_current(case_id, None if check_id is None else int(check_id))
        self.notify("case")
        return check

    def ingest_document(self, path: Path) -> dict[str, Any]:
        info = self.library.ingest(path)
        doc = self.cases.add_document(case_id=self.case_id, **info)
        self.notify("documents")
        return doc

    def remove_document(self, doc_id: int) -> None:
        doc = self.cases.remove_document(doc_id)
        if doc:
            self.library.delete(doc)
        self.notify("documents")

    def notify(self, what: str) -> None:
        if self.on_change is not None:
            self.on_change(what)


def build_repair_tools(ifixit_enabled: bool = True) -> list[dict[str, Any]]:
    box = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 4,
        "maxItems": 4,
        "description": "[x1,y1,x2,y2]，坐标约定见系统提示",
    }
    kind = {
        "type": "string",
        "enum": ["check", "fault", "ok", "danger"],
        "description": "check=待检查，fault=疑似故障，ok=已排除，danger=危险区域",
    }
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    point = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 2,
        "maxItems": 2,
        "description": "[x,y]，坐标约定同 box",
    }
    shape = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["rect", "circle", "point", "text", "arrow", "line", "probe"],
                "description": (
                    "rect=框住部件(box)；circle=圈出圆形部件或焊点(box)；point=精确点位(at)；"
                    "text=画面上的文字说明(at+text)；arrow=动作方向，从 from 指向 to，如撬、拔、推、拧的方向；"
                    "line=两点之间的关系或走线(from,to)；probe=万用表表笔位置(red 必填，black 可选)"
                ),
            },
            "label": {"type": "string", "description": "很短的标签，如 U3、C12、螺丝1"},
            "kind": kind,
            "note": {"type": "string", "description": "可选，一句话说明"},
            "box": box,
            "at": point,
            "text": {"type": "string", "description": "type=text 时显示的文字"},
            "from": point,
            "to": point,
            "red": point,
            "black": point,
            "meter": {"type": "string", "description": "probe 用：档位，如 DCV 20V、蜂鸣档、Ω 2k"},
            "expected": {"type": "string", "description": "probe 用：预期读数，如 3.3V、<1Ω"},
            "check_id": {"type": "integer", "description": "可选，对应的检查项"},
        },
        "required": ["type"],
    }
    shapes = {"type": "array", "items": shape}
    board_box = {
        **box,
        "description": "可选，被维修的板子/设备在画面里的范围 [x1,y1,x2,y2]，用于跟踪定位",
    }
    zoom_id = {
        "type": "integer",
        "description": "可选。坐标是在 zoom_view 返回的放大图上量的，就填它的 zoom_id；不填表示按原图",
    }

    def fn(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }

    tools = [
        fn("get_repair_case", "读取当前维修单：设备、现象、检查清单与结果、结论、资料列表、画面标注", empty),
        fn(
            "update_repair_case",
            "记录或更新设备型号和故障现象；没有维修单时会新建",
            {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "description": "设备品牌型号或板子名称"},
                    "symptom": {"type": "string", "description": "故障现象与发生经过"},
                },
            },
        ),
        fn(
            "propose_checklist",
            "生成检查清单并显示在侧栏；有 box 的项会同时标在画面上",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "短标题，如「测 3.3V 输出」"},
                                "instruction": {"type": "string", "description": "怎么测：档位、表笔位置、注意事项"},
                                "expected": {"type": "string", "description": "预期读数或现象"},
                                "box": box,
                                "kind": kind,
                            },
                            "required": ["title", "instruction"],
                        },
                    },
                    "replace_pending": {
                        "type": "boolean",
                        "description": "true 时先删掉还没查的旧项",
                    },
                    "zoom_id": zoom_id,
                },
                "required": ["items"],
            },
        ),
        fn(
            "zoom_view",
            "把附图的一块区域按原始分辨率裁出来放大给你看。元件小、丝印看不清、相似元件挨在一起时，先放大认清再标注",
            {
                "type": "object",
                "properties": {
                    "box": {**box, "description": "要放大的区域 [x1,y1,x2,y2]，按原图坐标"},
                    "reason": {"type": "string", "description": "想看清什么，如「读 U3 丝印」"},
                },
                "required": ["box"],
            },
        ),
        fn(
            "focus_check",
            "把某个检查项设为当前项，画面顶部显示它并加粗它的标注",
            {
                "type": "object",
                "properties": {"check_id": {"type": "integer"}},
                "required": ["check_id"],
            },
        ),
        fn(
            "record_check_result",
            "记录用户回报的检查结果；返回下一个待查项",
            {
                "type": "object",
                "properties": {
                    "check_id": {"type": "integer"},
                    "status": {"type": "string", "enum": ["pass", "fail", "skip", "pending"]},
                    "measured": {"type": "string", "description": "实测值，如 0.2V、短路、无波形"},
                    "note": {"type": "string"},
                },
                "required": ["check_id", "status"],
            },
        ),
        fn(
            "annotate",
            "在摄像头画面上画 AR 图形：框、圈、点、文字、箭头、连线、表笔位置。图形会贴在板子上跟着移动",
            {
                "type": "object",
                "properties": {
                    "items": shapes,
                    "replace": {"type": "boolean", "description": "true 时先清掉旧标注"},
                    "board_box": board_box,
                    "zoom_id": zoom_id,
                },
                "required": ["items"],
            },
        ),
        fn("clear_annotations", "清除画面上的全部维修标注（不影响指引步骤里的图形）", empty),
        fn(
            "start_guide",
            "开始一个分步操作指引（拆机、测量、更换元件等）。每步只显示自己的图形，用户做完后由你看画面确认",
            {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "如「拆下后盖」「测 3.3V 电源」"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string", "description": "短标题，如「拧下右上角螺丝」"},
                                "instruction": {"type": "string", "description": "具体怎么做，一两句"},
                                "verify": {"type": "string", "description": "做完后画面上应该看到什么，用于确认"},
                                "shapes": {**shapes, "description": "这一步要画的图形，最多 3 个"},
                            },
                            "required": ["title", "instruction"],
                        },
                    },
                    "board_box": {**board_box, "description": board_box["description"] + "，按原图坐标"},
                    "check_id": {"type": "integer", "description": "可选，这个指引对应的检查项"},
                    "zoom_id": zoom_id,
                },
                "required": ["title", "steps"],
            },
        ),
        fn(
            "set_guide_step",
            "跳到指引的第几步（从 1 开始）",
            {
                "type": "object",
                "properties": {"step": {"type": "integer"}},
                "required": ["step"],
            },
        ),
        fn(
            "update_guide_step",
            "修改某一步：重画图形或改说明。坐标基于本次附图",
            {
                "type": "object",
                "properties": {
                    "step": {"type": "integer", "description": "从 1 开始，省略时为当前步"},
                    "shapes": shapes,
                    "replace_shapes": {"type": "boolean", "description": "默认 true，替换这一步原有图形"},
                    "instruction": {"type": "string"},
                    "verify": {"type": "string"},
                    "zoom_id": zoom_id,
                },
            },
        ),
        fn(
            "confirm_step",
            "看附图判断当前步骤是否完成。通过会自动进入下一步；不通过时说明哪里不对，并可用 shapes 标出纠正位置",
            {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "feedback": {"type": "string", "description": "一句话：看到了什么、还差什么"},
                    "shapes": {**shapes, "description": "可选，不通过时标出纠正位置"},
                    "zoom_id": zoom_id,
                },
                "required": ["passed", "feedback"],
            },
        ),
        fn(
            "end_guide",
            "结束当前指引（全部完成或放弃）",
            {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "cancelled": {"type": "boolean", "description": "true 表示放弃"},
                },
            },
        ),
        fn(
            "search_documents",
            "在用户上传的原理图、维修手册里按关键词检索，返回文档、页码和片段。位号、网络名、芯片型号最好用",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "如「U3 VCC_3V3」「保险丝 F1」"},
                    "limit": {"type": "integer", "description": "最多返回几条，默认 6"},
                },
                "required": ["query"],
            },
        ),
        fn(
            "view_document_page",
            "查看资料某一页（或页内局部放大）的图片，用来读原理图",
            {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "integer"},
                    "page": {"type": "integer", "description": "从 1 开始"},
                    "region": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "可选，页内 0 到 1 的相对范围 [x1,y1,x2,y2]，用于放大局部",
                    },
                },
                "required": ["doc_id", "page"],
            },
        ),
        fn(
            "conclude",
            "给出诊断结论和处置步骤，显示在侧栏结论卡片里",
            {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "actions": {"type": "array", "items": {"type": "string"}, "description": "按顺序的处置步骤"},
                    "safety_notes": {"type": "string"},
                },
                "required": ["diagnosis", "actions"],
            },
        ),
    ]
    if ifixit_enabled:
        tools += [
            fn(
                "ifixit_search",
                "在 iFixit 搜索维修指南或故障排查页面。iFixit 以英文为主，用英文设备名和部件名搜，如 iPhone 12 battery",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "kind": {"type": "string", "enum": ["guide", "wiki"], "description": "guide=拆机维修指南，wiki=故障排查等页面"},
                    },
                    "required": ["query"],
                },
            ),
            fn(
                "ifixit_get_guide",
                "读取一篇 iFixit 指南的工具、配件和步骤文字。步骤多时用 start_step 分段读",
                {
                    "type": "object",
                    "properties": {
                        "guide_id": {"type": "integer"},
                        "start_step": {"type": "integer", "description": "从第几步开始，默认 1，每次最多 12 步"},
                    },
                    "required": ["guide_id"],
                },
            ),
            fn(
                "ifixit_device",
                "读取 iFixit 设备页：简介、相关指南列表和常见故障排查页",
                {
                    "type": "object",
                    "properties": {"device": {"type": "string", "description": "英文设备名，如 iPhone 12、PlayStation 4"}},
                    "required": ["device"],
                },
            ),
        ]
    return tools


def build_repair_handlers(ctx: RepairContext) -> dict[str, Callable[..., str]]:
    def get_repair_case() -> str:
        return _dump(
            {
                "case": ctx.case(),
                "documents": [
                    {k: d[k] for k in ("id", "name", "kind", "pages", "has_text")}
                    for d in ctx.documents()
                ],
                "annotations": [shape_summary(s) for s in ctx.annotations],
                "guide": _guide_summary(ctx.guide),
            }
        )

    def shapes_from(items: Any, zoom_id: Any = None) -> tuple[list[dict[str, Any]], int]:
        return ctx.make_shapes(items, zoom_id)

    def board(value: Any) -> list[float] | None:
        return ctx.normalize_box(value) if value else None

    def zoom_view(box: list[float], reason: str = "") -> str:
        region = ctx.normalize_box(box)
        if region is None:
            return "box 坐标无效，没有放大"
        zoom_id, caption, url = ctx.zoom(region)
        ctx.queue_image(caption, url)
        return (
            f"已附上放大图 zoom_id={zoom_id}。在放大图上量坐标时，用同样的坐标约定但以放大图本身为整张图，"
            f"并在 annotate / start_guide / update_guide_step / confirm_step / propose_checklist 里带 zoom_id={zoom_id}，"
            "程序会换算回原图。不带 zoom_id 时坐标按原图。"
        )

    def update_repair_case(device: str = "", symptom: str = "") -> str:
        case_id = ctx.ensure_case()
        case = ctx.cases.update_case(case_id, device=device, symptom=symptom)
        ctx.notify("case")
        return _dump({"case_id": case_id, "device": case["device"], "symptom": case["symptom"]})

    def propose_checklist(
        items: list[dict[str, Any]], replace_pending: bool = False, zoom_id: Any = None
    ) -> str:
        case_id = ctx.ensure_case()
        ctx.zoom_info(zoom_id)
        prepared = []
        boxed = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["box"] = ctx.normalize_box(item.get("box"), zoom_id) if item.get("box") else None
            prepared.append(entry)
        added = ctx.cases.add_checks(case_id, prepared, replace_pending=bool(replace_pending))
        new_annotations = []
        for check in added:
            if check["box"]:
                shape = make_shape(
                    {
                        "type": "rect",
                        "label": f"{check['seq']}. {check['title']}"[:24],
                        "kind": check["kind"],
                        "box": check["box"],
                        "note": check["expected"][:60],
                        "check_id": check["id"],
                    },
                    "norm1",
                )
                if shape:
                    boxed += 1
                    new_annotations.append(shape)
        if replace_pending:
            keep_ids = {c["id"] for c in (ctx.case() or {}).get("checks", [])}
            ctx.set_annotations(
                [a for a in ctx.annotations if a.get("check_id") is None or a.get("check_id") in keep_ids]
            )
        if new_annotations:
            ctx.add_annotations(new_annotations)
        ctx.notify("case")
        return _dump(
            {
                "added": [{"id": c["id"], "seq": c["seq"], "title": c["title"]} for c in added],
                "marked_on_screen": boxed,
            }
        )

    def focus_check(check_id: int) -> str:
        check = ctx.focus(check_id)
        return _dump({"current": check})

    def record_check_result(
        check_id: int, status: str, measured: str = "", note: str = ""
    ) -> str:
        return _dump(ctx.record_result(check_id, status, measured, note))

    def annotate(
        items: list[dict[str, Any]],
        replace: bool = False,
        board_box: Any = None,
        zoom_id: Any = None,
    ) -> str:
        made, skipped = shapes_from(items, zoom_id)
        if board_box:
            ctx.set_board_box(board(board_box))
        if made or replace:
            ctx.add_annotations(made, replace=bool(replace))
        result: dict[str, Any] = {"drawn": [f"{s['type']} {s['label']}".strip() for s in made]}
        if skipped:
            result["skipped"] = f"{skipped} 个图形坐标无效或缺字段，没有画出"
        if len(ctx.annotations) >= MAX_ANNOTATIONS:
            result["note"] = f"画面最多保留 {MAX_ANNOTATIONS} 个标注，旧的已被挤掉"
        return _dump(result)

    def clear_annotations() -> str:
        ctx.set_annotations([])
        return "已清除画面标注"

    def start_guide(
        title: str,
        steps: list[dict[str, Any]],
        board_box: Any = None,
        check_id: int | None = None,
        zoom_id: Any = None,
    ) -> str:
        prepared, report = [], []
        for i, step in enumerate(steps or []):
            if not isinstance(step, dict):
                continue
            made, skipped = shapes_from(step.get("shapes"), zoom_id)
            prepared.append({**step, "shapes": made[:MAX_STEP_SHAPES]})
            entry: dict[str, Any] = {"step": i + 1, "title": step.get("title"), "shapes": len(made)}
            if skipped:
                entry["skipped"] = skipped
            report.append(entry)
        guide = ctx.start_guide(title, prepared, check_id=check_id, board_box=board(board_box))
        return _dump(
            {
                "guide_id": guide["id"],
                "steps": report,
                "current_step": 1,
                "tip": "画面只显示当前步的图形。用户说做好了时，看附图调用 confirm_step",
            }
        )

    def set_guide_step(step: int) -> str:
        guide = ctx.goto_step(int(step) - 1)
        if guide is None:
            return "当前没有进行中的指引"
        return _dump(_guide_summary(guide))

    def update_guide_step(
        step: int | None = None,
        shapes: list[dict[str, Any]] | None = None,
        replace_shapes: bool = True,
        instruction: str | None = None,
        verify: str | None = None,
        zoom_id: Any = None,
    ) -> str:
        made = None
        skipped = 0
        if shapes is not None:
            made, skipped = shapes_from(shapes, zoom_id)
        guide = ctx.update_step(
            None if step is None else int(step) - 1,
            shapes=made,
            replace_shapes=bool(replace_shapes),
            instruction=instruction,
            verify=verify,
        )
        result: dict[str, Any] = {"updated": True, "guide": _guide_summary(guide)}
        if skipped:
            result["skipped"] = f"{skipped} 个图形坐标无效或缺字段"
        return _dump(result)

    def confirm_step(
        passed: bool,
        feedback: str = "",
        shapes: list[dict[str, Any]] | None = None,
        zoom_id: Any = None,
    ) -> str:
        made, _skipped = shapes_from(shapes, zoom_id)
        result = ctx.confirm(bool(passed), feedback, made)
        if "guide" in result:
            result["guide"] = _guide_summary(result["guide"])
        if result.get("finished"):
            result["tip"] = "指引已全部完成，告诉用户结果和下一步"
        return _dump(result)

    def end_guide(summary: str = "", cancelled: bool = False) -> str:
        guide = ctx.end_guide(summary, "cancelled" if cancelled else "done")
        return "当前没有进行中的指引" if guide is None else "指引已结束"

    def search_documents(query: str, limit: int = 6) -> str:
        docs = ctx.documents()
        if not docs:
            return "用户还没有上传资料"
        hits = ctx.library.search(docs, query, limit=max(1, min(int(limit or 6), 12)))
        if not hits:
            scanned = [d["id"] for d in docs if not d["has_text"]]
            tip = "没有文字命中。"
            if scanned:
                tip += f"文档 {scanned} 没有可检索文字（扫描件或图片），可以用 view_document_page 直接看。"
            return tip
        return _dump(hits)

    def view_document_page(doc_id: int, page: int, region: list[float] | None = None) -> str:
        doc = ctx.cases.get_document(int(doc_id))
        if doc is None:
            return f"资料不存在: {doc_id}"
        caption, url = ctx.library.render(doc, int(page), region, max_side=ctx.doc_max_side)
        ctx.queue_image(caption, url)
        text = ctx.library.page_texts(doc)
        page_text = text[int(page) - 1] if 0 < int(page) <= len(text) else ""
        result = f"已附上图片：{caption}。"
        if page_text.strip():
            result += "本页文字（节选）：" + " ".join(page_text.split())[:1500]
        return result

    def conclude(diagnosis: str, actions: list[str], safety_notes: str = "") -> str:
        case_id = ctx.ensure_case()
        ctx.cases.conclude(case_id, diagnosis, actions, safety_notes)
        ctx.notify("conclusion")
        return "结论已显示在侧栏"

    def ifixit_search(query: str, kind: str = "guide") -> str:
        if ctx.ifixit is None:
            return "iFixit 查询已关闭"
        return _dump(ctx.ifixit.search(query, kind=kind))

    def ifixit_get_guide(guide_id: int, start_step: int = 1) -> str:
        if ctx.ifixit is None:
            return "iFixit 查询已关闭"
        return _dump(ctx.ifixit.guide(guide_id, start_step=start_step, max_steps=12))

    def ifixit_device(device: str) -> str:
        if ctx.ifixit is None:
            return "iFixit 查询已关闭"
        return _dump(ctx.ifixit.device(device))

    def logged(name: str, fn: Callable[..., str]) -> Callable[..., str]:
        def call(**kw: Any) -> str:
            try:
                result = fn(**kw)
            except Exception as exc:
                if ctx.debug is not None:
                    ctx.debug.tool(name, kw, f"ERROR {type(exc).__name__}: {exc}")
                raise
            if ctx.debug is not None:
                ctx.debug.tool(name, kw, result)
            return result

        return call

    handlers: dict[str, Callable[..., str]] = {
        "get_repair_case": lambda **_: get_repair_case(),
        "update_repair_case": lambda **kw: update_repair_case(**kw),
        "propose_checklist": lambda **kw: propose_checklist(**kw),
        "zoom_view": lambda **kw: zoom_view(**kw),
        "focus_check": lambda **kw: focus_check(**kw),
        "record_check_result": lambda **kw: record_check_result(**kw),
        "annotate": lambda **kw: annotate(**kw),
        "clear_annotations": lambda **_: clear_annotations(),
        "start_guide": lambda **kw: start_guide(**kw),
        "set_guide_step": lambda **kw: set_guide_step(**kw),
        "update_guide_step": lambda **kw: update_guide_step(**kw),
        "confirm_step": lambda **kw: confirm_step(**kw),
        "end_guide": lambda **kw: end_guide(**kw),
        "search_documents": lambda **kw: search_documents(**kw),
        "view_document_page": lambda **kw: view_document_page(**kw),
        "conclude": lambda **kw: conclude(**kw),
        "ifixit_search": lambda **kw: ifixit_search(**kw),
        "ifixit_get_guide": lambda **kw: ifixit_get_guide(**kw),
        "ifixit_device": lambda **kw: ifixit_device(**kw),
    }
    return {name: logged(name, fn) for name, fn in handlers.items()}
