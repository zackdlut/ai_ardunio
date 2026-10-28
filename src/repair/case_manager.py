from __future__ import annotations

import json
from typing import Any, Callable

from sqlalchemy import or_, select

from project.models import get_session_factory
from repair.models import RepairCase, RepairCheck, RepairDocument

CHECK_STATUSES = ("pending", "pass", "fail", "skip")
CHECK_KINDS = ("check", "fault", "ok", "danger")


def _loads_list(text: str) -> list:
    try:
        value = json.loads(text or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _check_dict(row: RepairCheck) -> dict[str, Any]:
    box = _loads_list(row.box)
    return {
        "id": row.id,
        "seq": row.seq,
        "title": row.title,
        "instruction": row.instruction,
        "expected": row.expected,
        "box": box if len(box) == 4 else None,
        "kind": row.kind,
        "status": row.status,
        "measured": row.measured,
        "note": row.note,
    }


def _doc_dict(row: RepairDocument) -> dict[str, Any]:
    return {
        "id": row.id,
        "case_id": row.case_id,
        "name": row.name,
        "path": row.path,
        "kind": row.kind,
        "pages": row.pages,
        "has_text": row.has_text,
    }


class CaseManager:
    """Repair cases, their checklist and uploaded documents.

    Every call opens its own short session, so the LLM worker thread and the
    GUI thread can both use one instance.
    """

    def __init__(self, session_factory: Callable | None = None) -> None:
        self._factory = session_factory or get_session_factory()

    def create_case(self, device: str = "", symptom: str = "") -> dict[str, Any]:
        with self._factory() as db:
            row = RepairCase(device=device.strip(), symptom=symptom.strip())
            db.add(row)
            db.commit()
            return self._case_dict(db, row)

    def latest_open_case(self) -> dict[str, Any] | None:
        with self._factory() as db:
            row = db.scalars(
                select(RepairCase)
                .where(RepairCase.status == "open")
                .order_by(RepairCase.id.desc())
                .limit(1)
            ).first()
            return self._case_dict(db, row) if row else None

    def get_case(self, case_id: int) -> dict[str, Any] | None:
        with self._factory() as db:
            row = db.get(RepairCase, case_id)
            return self._case_dict(db, row) if row else None

    def update_case(
        self,
        case_id: int,
        device: str | None = None,
        symptom: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        with self._factory() as db:
            row = self._require_case(db, case_id)
            if device is not None and device.strip():
                row.device = device.strip()
            if symptom is not None and symptom.strip():
                row.symptom = symptom.strip()
            if status in ("open", "closed"):
                row.status = status
            db.commit()
            return self._case_dict(db, row)

    def add_checks(
        self,
        case_id: int,
        items: list[dict[str, Any]],
        replace_pending: bool = False,
    ) -> list[dict[str, Any]]:
        with self._factory() as db:
            case = self._require_case(db, case_id)
            existing = list(
                db.scalars(select(RepairCheck).where(RepairCheck.case_id == case_id))
            )
            if replace_pending:
                for row in existing:
                    if row.status == "pending":
                        db.delete(row)
                existing = [row for row in existing if row.status != "pending"]
                if case.current_check_id not in {row.id for row in existing}:
                    case.current_check_id = None
            seq = max((row.seq for row in existing), default=0)
            added: list[RepairCheck] = []
            for item in items:
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                seq += 1
                kind = str(item.get("kind") or "check")
                box = item.get("box")
                row = RepairCheck(
                    case_id=case_id,
                    seq=seq,
                    title=title,
                    instruction=str(item.get("instruction") or "").strip(),
                    expected=str(item.get("expected") or "").strip(),
                    box=json.dumps(box) if box else "",
                    kind=kind if kind in CHECK_KINDS else "check",
                )
                db.add(row)
                added.append(row)
            db.flush()
            if case.current_check_id is None and added:
                case.current_check_id = added[0].id
            db.commit()
            return [_check_dict(row) for row in added]

    def set_current(self, case_id: int, check_id: int | None) -> dict[str, Any] | None:
        with self._factory() as db:
            case = self._require_case(db, case_id)
            if check_id is None:
                case.current_check_id = None
                db.commit()
                return None
            row = self._require_check(db, case_id, check_id)
            case.current_check_id = row.id
            db.commit()
            return _check_dict(row)

    def record_result(
        self,
        case_id: int,
        check_id: int,
        status: str,
        measured: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        if status not in CHECK_STATUSES:
            raise ValueError(f"status 只能是 {', '.join(CHECK_STATUSES)}")
        with self._factory() as db:
            row = self._require_check(db, case_id, check_id)
            row.status = status
            if measured.strip():
                row.measured = measured.strip()
            if note.strip():
                row.note = note.strip()
            db.commit()
            return _check_dict(row)

    def next_pending(self, case_id: int, after_check_id: int | None = None) -> dict[str, Any] | None:
        with self._factory() as db:
            rows = list(
                db.scalars(
                    select(RepairCheck)
                    .where(RepairCheck.case_id == case_id)
                    .order_by(RepairCheck.seq)
                )
            )
            after_seq = -1
            if after_check_id is not None:
                after_seq = next((r.seq for r in rows if r.id == after_check_id), -1)
            pending = [r for r in rows if r.status == "pending"]
            for row in pending:
                if row.seq > after_seq:
                    return _check_dict(row)
            return _check_dict(pending[0]) if pending else None

    def conclude(
        self,
        case_id: int,
        diagnosis: str,
        actions: list[str] | None = None,
        safety_notes: str = "",
    ) -> dict[str, Any]:
        with self._factory() as db:
            row = self._require_case(db, case_id)
            row.diagnosis = diagnosis.strip()
            row.actions = json.dumps(
                [str(a).strip() for a in (actions or []) if str(a).strip()],
                ensure_ascii=False,
            )
            row.safety_notes = safety_notes.strip()
            db.commit()
            return self._case_dict(db, row)

    def add_document(
        self,
        name: str,
        path: str,
        kind: str,
        pages: int,
        has_text: bool,
        case_id: int | None = None,
    ) -> dict[str, Any]:
        with self._factory() as db:
            row = RepairDocument(
                case_id=case_id,
                name=name,
                path=path,
                kind=kind,
                pages=max(1, int(pages)),
                has_text=bool(has_text),
            )
            db.add(row)
            db.commit()
            return _doc_dict(row)

    def get_document(self, doc_id: int) -> dict[str, Any] | None:
        with self._factory() as db:
            row = db.get(RepairDocument, doc_id)
            return _doc_dict(row) if row else None

    def list_documents(self, case_id: int | None = None) -> list[dict[str, Any]]:
        """Documents of ``case_id`` plus the shared ones (uploaded without a case)."""
        with self._factory() as db:
            query = select(RepairDocument).order_by(RepairDocument.id)
            if case_id is not None:
                query = query.where(
                    or_(RepairDocument.case_id == case_id, RepairDocument.case_id.is_(None))
                )
            return [_doc_dict(row) for row in db.scalars(query)]

    def remove_document(self, doc_id: int) -> dict[str, Any] | None:
        with self._factory() as db:
            row = db.get(RepairDocument, doc_id)
            if row is None:
                return None
            info = _doc_dict(row)
            db.delete(row)
            db.commit()
            return info

    def _require_case(self, db, case_id: int) -> RepairCase:
        row = db.get(RepairCase, int(case_id))
        if row is None:
            raise ValueError(f"维修单不存在: {case_id}")
        return row

    def _require_check(self, db, case_id: int, check_id: int) -> RepairCheck:
        row = db.get(RepairCheck, int(check_id))
        if row is None or row.case_id != int(case_id):
            raise ValueError(f"检查项不存在: {check_id}")
        return row

    def _case_dict(self, db, row: RepairCase) -> dict[str, Any]:
        checks = db.scalars(
            select(RepairCheck).where(RepairCheck.case_id == row.id).order_by(RepairCheck.seq)
        )
        return {
            "id": row.id,
            "device": row.device,
            "symptom": row.symptom,
            "status": row.status,
            "diagnosis": row.diagnosis,
            "actions": _loads_list(row.actions),
            "safety_notes": row.safety_notes,
            "current_check_id": row.current_check_id,
            "checks": [_check_dict(c) for c in checks],
        }
