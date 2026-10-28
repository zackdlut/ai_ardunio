"""A repair case runs from symptom to checklist, results and conclusion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import repair.models  # noqa: E402,F401
from agent.repair_context import build_repair_extra_system  # noqa: E402
from agent.repair_tools import (  # noqa: E402
    RepairContext,
    build_repair_handlers,
    build_repair_tools,
)
from project.models import Base  # noqa: E402
from repair.case_manager import CaseManager  # noqa: E402
from repair.documents import DocumentLibrary  # noqa: E402


@pytest.fixture()
def cases() -> CaseManager:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return CaseManager(sessionmaker(bind=engine, expire_on_commit=False))


def test_checklist_flow(cases: CaseManager) -> None:
    case = cases.create_case("电源板", "不开机")
    added = cases.add_checks(
        case["id"],
        [
            {"title": "看保险丝", "instruction": "目视", "box": [0.1, 0.1, 0.2, 0.2]},
            {"title": "测 12V", "instruction": "直流 20V 档"},
            {"title": "", "instruction": "空标题会被忽略"},
        ],
    )
    assert [c["seq"] for c in added] == [1, 2]
    assert cases.get_case(case["id"])["current_check_id"] == added[0]["id"]
    assert added[0]["box"] == [0.1, 0.1, 0.2, 0.2] and added[1]["box"] is None

    cases.record_result(case["id"], added[0]["id"], "pass")
    assert cases.next_pending(case["id"], added[0]["id"])["id"] == added[1]["id"]
    with pytest.raises(ValueError):
        cases.record_result(case["id"], added[1]["id"], "broken")
    with pytest.raises(ValueError):
        cases.record_result(case["id"] + 1, added[1]["id"], "pass")

    more = cases.add_checks(case["id"], [{"title": "测 5V"}], replace_pending=True)
    titles = [c["title"] for c in cases.get_case(case["id"])["checks"]]
    assert titles == ["看保险丝", "测 5V"] and more[0]["seq"] == 2

    done = cases.conclude(case["id"], "D1 击穿", ["更换 D1 SS34", ""], "先断电")
    assert done["actions"] == ["更换 D1 SS34"]
    assert cases.latest_open_case()["id"] == case["id"]
    cases.update_case(case["id"], status="closed")
    assert cases.latest_open_case() is None


def test_tools_drive_case_overlay_and_documents(cases: CaseManager, tmp_path: Path) -> None:
    events: list[str] = []
    ctx = RepairContext(
        cases=cases,
        library=DocumentLibrary(tmp_path / "docs"),
        image_size=(1280, 720),
        on_change=events.append,
    )
    tools = build_repair_handlers(ctx)

    assert json.loads(tools["update_repair_case"](device="路由器", symptom="灯不亮"))["case_id"] == 1
    result = json.loads(
        tools["propose_checklist"](
            items=[
                {"title": "测适配器输出", "instruction": "直流 20V 档", "expected": "12V", "box": [0, 0, 300, 300]},
                {"title": "看 DC 座焊点", "instruction": "放大看"},
            ]
        )
    )
    assert result["marked_on_screen"] == 1
    first, second = (item["id"] for item in result["added"])
    assert ctx.annotations[0]["check_id"] == first
    assert ctx.annotations[0]["pts"] == [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]

    recorded = json.loads(tools["record_check_result"](check_id=first, status="fail", measured="0V"))
    assert recorded["next_check"]["id"] == second
    assert ctx.annotations[0]["kind"] == "fault" and "0V" in ctx.annotations[0]["note"]
    assert ctx.case()["current_check_id"] == second

    drawn = json.loads(tools["annotate"](items=[{"label": "C3", "box": [400, 400, 500, 500]}, {"label": "x", "box": "?"}]))
    assert drawn["drawn"] == ["rect C3"] and "skipped" in drawn
    assert "已清除" in tools["clear_annotations"]()
    assert ctx.annotations == []

    assert "还没有上传" in tools["search_documents"](query="U3")
    pdf = tmp_path / "manual.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Adapter jack J1 feeds U3 MP1584 buck")
    doc.save(str(pdf))
    stored = ctx.ingest_document(pdf)
    assert json.loads(tools["search_documents"](query="MP1584"))[0]["doc_id"] == stored["id"]
    text = tools["view_document_page"](doc_id=stored["id"], page=1)
    assert "已附上图片" in text and "MP1584" in text
    images = ctx.drain_images()
    assert len(images) == 1 and images[0][1].startswith("data:image/jpeg;base64,")
    assert ctx.drain_images() == []
    assert "不存在" in tools["view_document_page"](doc_id=999, page=1)

    assert "结论" in tools["conclude"](diagnosis="适配器损坏", actions=["更换 12V 1A 适配器"])
    assert ctx.case()["diagnosis"] == "适配器损坏"
    assert {"case", "annotations", "documents", "conclusion"} <= set(events)

    extra = build_repair_extra_system(
        ctx.case(), ctx.documents(), ctx.annotations, [], False, has_image=True, frozen=True
    )
    assert "路由器" in extra and "[异常] 实测: 0V" in extra and "← 当前项" in extra
    assert "0 到 1000" in extra and "已定格" in extra

    ctx.new_case("新设备")
    assert ctx.case()["device"] == "新设备" and cases.get_case(1)["status"] == "closed"


def test_ifixit_tools_follow_the_setting(cases: CaseManager, tmp_path: Path) -> None:
    names = {t["function"]["name"] for t in build_repair_tools(ifixit_enabled=False)}
    assert "annotate" in names and "ifixit_search" not in names
    names = {t["function"]["name"] for t in build_repair_tools(ifixit_enabled=True)}
    assert {"ifixit_search", "ifixit_get_guide", "ifixit_device"} <= names

    ctx = RepairContext(cases=cases, library=DocumentLibrary(tmp_path))
    assert "已关闭" in build_repair_handlers(ctx)["ifixit_search"](query="x")
