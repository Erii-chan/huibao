"""长期记忆存储：单文件 JSONL + 字符二元组检索，零第三方依赖。

和「笔记本」的区别：笔记本记的是查来的知识（带来源），这里记的是
关于这个人的长期事实/偏好/计划，自动从对话里抽，自动带进上下文。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid

_PUNCT = re.compile(
    r"[\s\u3000,，.。!！?？;；:：、\"'“”‘’()（）\[\]【】<>《》~～…—\-_/\\|*#`]+"
)

KINDS = ("事实", "偏好", "计划", "事件", "关系")


def normalize(text: str) -> str:
    return _PUNCT.sub("", (text or "").lower())


def grams(text: str) -> set[str]:
    t = normalize(text)
    if not t:
        return set()
    if len(t) == 1:
        return {t}
    return {t[i : i + 2] for i in range(len(t) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._items: list[dict] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.load()

    # ---------------- 持久化 ----------------

    def load(self) -> None:
        items: list[dict] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj.get("id") and obj.get("text"):
                        items.append(obj)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        with self._lock:
            self._items = items

    def _save_locked(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for item in self._items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)

    # ---------------- 读写 ----------------

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def add(
        self,
        text: str,
        kind: str = "事实",
        source: str = "auto",
        weight: float = 1.0,
        dedup: float = 0.75,
    ) -> dict | None:
        """写入一条；和已有条目太像就不重复写，只刷新时间与权重。"""
        text = (text or "").strip()
        if not text:
            return None
        if len(text) > 200:
            text = text[:200]
        g = grams(text)
        now = time.time()
        with self._lock:
            for item in self._items:
                if jaccard(g, grams(str(item.get("text")))) >= dedup:
                    item["lu"] = now
                    item["weight"] = round(min(float(item.get("weight") or 1.0) + 0.2, 3.0), 2)
                    self._save_locked()
                    return item
            item = {
                "id": uuid.uuid4().hex[:8],
                "text": text,
                "kind": kind if kind in KINDS else "事实",
                "ts": now,
                "lu": now,
                "weight": round(float(weight), 2),
                "hits": 0,
                "src": source,
            }
            self._items.append(item)
            self._prune_locked()
            self._save_locked()
            return item

    def _prune_locked(self, limit: int = 1200) -> None:
        if len(self._items) <= limit:
            return
        # 超量时优先丢掉「权重低 + 老 + 从没被召回」的
        self._items.sort(
            key=lambda it: (float(it.get("weight") or 1.0), float(it.get("lu") or 0)),
            reverse=True,
        )
        del self._items[limit:]

    def remove(self, key: str) -> int:
        """按 id 或关键词删，返回删掉的条数。"""
        key = (key or "").strip()
        if not key:
            return 0
        kg = grams(key)
        with self._lock:
            keep, removed = [], 0
            for item in self._items:
                hit = item.get("id") == key or (
                    kg and jaccard(kg, grams(str(item.get("text")))) >= 0.3
                )
                if hit:
                    removed += 1
                else:
                    keep.append(item)
            if removed:
                self._items = keep
                self._save_locked()
            return removed

    def touch(self, item_id: str) -> None:
        with self._lock:
            for item in self._items:
                if item.get("id") == item_id:
                    item["hits"] = int(item.get("hits") or 0) + 1
                    return

    def search(
        self, query: str, limit: int = 6, min_score: float = 0.3, recent: int = 0
    ) -> list[tuple[float, dict]]:
        """字符二元组 + 新鲜度打分。中文不需要分词，几百条量级够快。"""
        q = grams(query)
        now = time.time()
        scored: list[tuple[float, dict]] = []
        with self._lock:
            items = list(self._items)
        for item in items:
            ig = grams(str(item.get("text")))
            cover = (len(q & ig) / len(q)) if q else 0.0
            if cover < min_score:
                continue
            age_days = max((now - float(item.get("lu") or item.get("ts") or now)) / 86400, 0)
            fresh = math.exp(-age_days / 90)
            weight = float(item.get("weight") or 1.0)
            score = cover * 0.75 + fresh * 0.15 + min(weight, 3.0) / 3.0 * 0.10
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[:limit]
        if recent > 0:
            ids = {it.get("id") for _s, it in picked}
            latest = sorted(
                items, key=lambda it: float(it.get("lu") or 0), reverse=True
            )[:recent]
            for item in latest:
                if item.get("id") not in ids:
                    picked.append((0.0, item))
                    ids.add(item.get("id"))
        return picked

    def stats(self) -> dict:
        with self._lock:
            items = list(self._items)
        kinds: dict[str, int] = {}
        for item in items:
            k = str(item.get("kind") or "事实")
            kinds[k] = kinds.get(k, 0) + 1
        return {"total": len(items), "kinds": kinds}
