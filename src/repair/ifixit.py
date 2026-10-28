"""Read-only iFixit API v2.0 client. No key is needed for public guides."""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import quote

import httpx

_USER_AGENT = "ai-ardunio-repair-copilot/1.0"
_CACHE_TTL_S = 30 * 60
_MAX_STEPS = 40


class IFixitClient:
    """Every public method returns a dict; failures come back as ``{"error": ...}``."""

    def __init__(
        self,
        base_url: str = "https://www.ifixit.com/api/2.0",
        timeout_s: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        # trust_env picks up HTTPS_PROXY / NO_PROXY from .env, same as the rest of the app.
        self._client = client or httpx.Client(
            timeout=timeout_s,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
            follow_redirects=True,
            trust_env=True,
        )
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def search(self, query: str, kind: str = "guide", limit: int = 6) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "搜索词为空"}
        kind = kind if kind in ("guide", "wiki") else "guide"
        params = {"filter": kind, "limit": max(1, min(limit, 10))}
        data = self._get(f"/search/{quote(query, safe='')}", params)
        if "error" in data:
            return data
        results = []
        for item in data.get("results") or []:
            entry = {
                "type": item.get("dataType"),
                "title": item.get("title") or item.get("display_title"),
                "summary": _clip(item.get("summary") or item.get("text") or "", 200),
                "url": item.get("url"),
            }
            if item.get("guideid") is not None:
                entry.update(
                    guide_id=item.get("guideid"),
                    device=item.get("category"),
                    subject=item.get("subject"),
                    difficulty=item.get("difficulty"),
                )
            results.append(entry)
        return {"query": query, "total": data.get("totalResults", len(results)), "results": results}

    def guide(self, guide_id: int, start_step: int = 1, max_steps: int = _MAX_STEPS) -> dict[str, Any]:
        try:
            guide_id = int(guide_id)
        except (TypeError, ValueError):
            return {"error": f"guide_id 应为数字: {guide_id}"}
        data = self._get(f"/guides/{guide_id}")
        if "error" in data:
            return data
        first = max(1, int(start_step or 1)) - 1
        steps = []
        for step in (data.get("steps") or [])[first : first + max(1, max_steps)]:
            lines = [
                ("  " * int(line.get("level") or 0))
                + _bullet(line.get("bullet"))
                + str(line.get("text_raw") or "").strip()
                for line in step.get("lines") or []
                if str(line.get("text_raw") or "").strip()
            ]
            media = step.get("media") or {}
            images = [
                img.get("standard") or img.get("medium")
                for img in (media.get("data") or [])
                if isinstance(img, dict) and media.get("type") == "image"
            ]
            steps.append(
                {
                    "n": step.get("orderby"),
                    "title": step.get("title") or "",
                    "lines": lines,
                    "images": [url for url in images if url][:3],
                }
            )
        return {
            "guide_id": data.get("guideid"),
            "title": data.get("title"),
            "device": data.get("category"),
            "difficulty": data.get("difficulty"),
            "time_required": data.get("time_required"),
            "introduction": _clip(data.get("introduction_raw") or "", 800),
            "tools": [t.get("text") for t in data.get("tools") or [] if t.get("text")],
            "parts": [p.get("text") for p in data.get("parts") or [] if p.get("text")],
            "steps": steps,
            "total_steps": len(data.get("steps") or []),
            "url": data.get("url"),
            "source": "iFixit (CC BY-NC-SA)",
        }

    def device(self, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"error": "设备名为空"}
        data = self._get(f"/wikis/CATEGORY/{quote(name, safe='')}")
        if "error" in data:
            return data
        return {
            "device": data.get("display_title") or data.get("title"),
            "description": _clip(data.get("description") or "", 600),
            "guides": [
                {"guide_id": g.get("guideid"), "title": g.get("title"), "type": g.get("type")}
                for g in (data.get("guides") or [])[:20]
            ],
            "troubleshooting": [
                {"title": w.get("display_title") or w.get("title"), "summary": _clip(w.get("summary") or "", 160), "url": w.get("url")}
                for w in (data.get("related_wikis") or [])[:10]
            ],
            "answers_url": data.get("solutions_url"),
            "url": data.get("url") or f"https://www.ifixit.com/Device/{quote(name.replace(' ', '_'))}",
        }

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < _CACHE_TTL_S:
                return hit[1]
        try:
            response = self._client.get(self._base + path, params=params)
        except httpx.TimeoutException:
            return {"error": "iFixit 请求超时，请检查网络或 HTTPS_PROXY"}
        except httpx.HTTPError as exc:
            return {"error": f"无法连接 iFixit: {exc}（请检查 HTTPS_PROXY）"}
        if response.status_code == 404:
            return {"error": "iFixit 没有找到对应内容，可以换个英文名称再搜"}
        if response.status_code >= 400:
            return {"error": f"iFixit 返回 HTTP {response.status_code}"}
        try:
            data = response.json()
        except ValueError:
            return {"error": "iFixit 返回的不是 JSON"}
        if not isinstance(data, dict):
            data = {"results": data}
        with self._lock:
            self._cache[key] = (now, data)
        return data


def _bullet(kind: Any) -> str:
    return {"icon_caution": "[注意] ", "icon_note": "[提示] ", "icon_reminder": "[提醒] "}.get(
        str(kind or ""), "- "
    )


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
