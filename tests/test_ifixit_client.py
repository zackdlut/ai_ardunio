"""iFixit answers are trimmed for the model; failures come back as Chinese errors."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from repair.ifixit import IFixitClient  # noqa: E402

_GUIDE = {
    "guideid": 140588,
    "title": "iPhone 12 Battery Replacement",
    "category": "iPhone 12",
    "difficulty": "Moderate",
    "time_required": "1 - 2 hours",
    "introduction_raw": "Replace the battery.",
    "tools": [{"text": "P2 Pentalobe Screwdriver"}],
    "parts": [{"text": "iPhone 12 Battery"}],
    "url": "https://www.ifixit.com/Guide/x/140588",
    "steps": [
        {
            "orderby": n,
            "title": f"Step {n}",
            "lines": [
                {"text_raw": "Power off the phone.", "bullet": "black", "level": 0},
                {"text_raw": "Battery may be swollen.", "bullet": "icon_caution", "level": 1},
            ],
            "media": {"type": "image", "data": [{"standard": f"https://img/{n}.standard"}]},
        }
        for n in range(1, 21)
    ],
}


def _client(handler) -> tuple[IFixitClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.Client(transport=httpx.MockTransport(record))
    return IFixitClient("https://api.test/2.0", client=http), seen


def test_search_trims_results_and_caches() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.raw_path.startswith(b"/2.0/search/iPhone%2012%20battery?")
        assert request.url.params["filter"] == "guide"
        return httpx.Response(
            200,
            json={
                "totalResults": 1,
                "results": [
                    {
                        "dataType": "guide",
                        "guideid": 140588,
                        "title": "iPhone 12 Battery Replacement",
                        "category": "iPhone 12",
                        "subject": "Battery",
                        "difficulty": "Moderate",
                        "summary": "x" * 500,
                        "url": "https://www.ifixit.com/Guide/x/140588",
                        "image": {"huge": "..."},
                    }
                ],
            },
        )

    client, seen = _client(handler)
    result = client.search("iPhone 12 battery")
    hit = result["results"][0]
    assert hit["guide_id"] == 140588 and hit["device"] == "iPhone 12"
    assert len(hit["summary"]) <= 200 and "image" not in hit
    client.search("iPhone 12 battery")
    assert len(seen) == 1


def test_guide_steps_are_paged_and_flattened() -> None:
    client, _seen = _client(lambda request: httpx.Response(200, json=_GUIDE))
    guide = client.guide(140588, start_step=5, max_steps=3)
    assert [s["n"] for s in guide["steps"]] == [5, 6, 7]
    assert guide["total_steps"] == 20
    assert guide["steps"][0]["lines"] == ["- Power off the phone.", "  [注意] Battery may be swollen."]
    assert guide["steps"][0]["images"] == ["https://img/5.standard"]
    assert guide["tools"] == ["P2 Pentalobe Screwdriver"]
    assert "iFixit" in guide["source"]


def test_errors_do_not_raise() -> None:
    missing, _ = _client(lambda request: httpx.Response(404, json={"message": "Wiki not found"}))
    assert "没有找到" in missing.device("No Such Device")["error"]

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    slow, _ = _client(boom)
    assert "超时" in slow.search("anything")["error"]
    assert "数字" in slow.guide("abc")["error"]
    assert "为空" in slow.search("  ")["error"]
