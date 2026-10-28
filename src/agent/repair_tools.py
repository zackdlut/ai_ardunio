from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from repair.annotations import make_annotation, merge_annotations, normalize_box
from repair.case_manager import CaseManager
from repair.documents import DocumentLibrary
from repair.ifixit import IFixitClient

_STATUS_KIND = {"fail": "fault", "pass": "ok", "skip": "ok"}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


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
    _pending_images: list[tuple[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

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
        self.notify("case")
        self.notify("annotations")
        return case

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
                },
                "required": ["items"],
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
            "在摄像头画面上框出位置：检查点、疑似故障点、已排除点或危险区域",
            {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "很短的标签，如 U3、C12、保险丝"},
                                "box": box,
                                "kind": kind,
                                "note": {"type": "string", "description": "可选，一句话说明"},
                                "check_id": {"type": "integer", "description": "可选，对应的检查项"},
                            },
                            "required": ["label", "box"],
                        },
                    },
                    "replace": {"type": "boolean", "description": "true 时先清掉旧标注"},
                },
                "required": ["items"],
            },
        ),
        fn("clear_annotations", "清除画面上的全部维修标注", empty),
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
                "annotations": ctx.annotations,
            }
        )

    def update_repair_case(device: str = "", symptom: str = "") -> str:
        case_id = ctx.ensure_case()
        case = ctx.cases.update_case(case_id, device=device, symptom=symptom)
        ctx.notify("case")
        return _dump({"case_id": case_id, "device": case["device"], "symptom": case["symptom"]})

    def propose_checklist(items: list[dict[str, Any]], replace_pending: bool = False) -> str:
        case_id = ctx.ensure_case()
        prepared = []
        boxed = 0
        for item in items or []:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["box"] = normalize_box(item.get("box"), ctx.box_format, ctx.image_size)
            prepared.append(entry)
        added = ctx.cases.add_checks(case_id, prepared, replace_pending=bool(replace_pending))
        new_annotations = []
        for check in added:
            if check["box"]:
                boxed += 1
                new_annotations.append(
                    {
                        "label": f"{check['seq']}. {check['title']}"[:24],
                        "kind": check["kind"],
                        "box": check["box"],
                        "note": check["expected"][:60],
                        "check_id": check["id"],
                    }
                )
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

    def annotate(items: list[dict[str, Any]], replace: bool = False) -> str:
        made, skipped = [], 0
        for item in items or []:
            ann = make_annotation(item, ctx.box_format, ctx.image_size) if isinstance(item, dict) else None
            if ann is None:
                skipped += 1
                continue
            made.append(ann)
        if made or replace:
            ctx.add_annotations(made, replace=bool(replace))
        result = {"drawn": [a["label"] for a in made]}
        if skipped:
            result["skipped"] = f"{skipped} 个框坐标无效，没有画出"
        return _dump(result)

    def clear_annotations() -> str:
        ctx.set_annotations([])
        return "已清除画面标注"

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

    return {
        "get_repair_case": lambda **_: get_repair_case(),
        "update_repair_case": lambda **kw: update_repair_case(**kw),
        "propose_checklist": lambda **kw: propose_checklist(**kw),
        "focus_check": lambda **kw: focus_check(**kw),
        "record_check_result": lambda **kw: record_check_result(**kw),
        "annotate": lambda **kw: annotate(**kw),
        "clear_annotations": lambda **_: clear_annotations(),
        "search_documents": lambda **kw: search_documents(**kw),
        "view_document_page": lambda **kw: view_document_page(**kw),
        "conclude": lambda **kw: conclude(**kw),
        "ifixit_search": lambda **kw: ifixit_search(**kw),
        "ifixit_get_guide": lambda **kw: ifixit_get_guide(**kw),
        "ifixit_device": lambda **kw: ifixit_device(**kw),
    }
