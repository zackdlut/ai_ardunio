from __future__ import annotations

from project.models import ProjectSession, get_session_factory


def create_session(project_id: str) -> int:
    factory = get_session_factory()
    with factory() as db:
        row = ProjectSession(project_id=project_id, current_step_id="")
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id
