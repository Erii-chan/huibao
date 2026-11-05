"""绘宝的记忆：真正的长期记忆。

- 每轮对话后自动从对话里抽取「关于绘梨衣的长期事实」，写进本地 JSONL；
- 每次生成回复前，按本轮话题召回相关条目，拼进 system prompt；
- 绘宝也能自己调工具记/查/忘。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .store import KINDS, MemoryStore

PLUGIN_NAME = "astrbot_plugin_memorize"  # 目录/数据目录名，别改，改了记忆就找不到了
DISPLAY_NAME = "绘宝的记忆"

EXTRACT_PROMPT = """你是记忆整理器。下面是一轮对话（用户：绘梨衣，助手：绘宝）。
从用户的话里抽取值得长期记住的信息，只抽这几类：
- 偏好：明确说的喜欢/讨厌/习惯
- 事实：关于他自己的稳定信息（身份、称呼、作息、设备、常做的事）
- 计划：说了要做的、约好的事
- 事件：发生了的、以后可能要提起的事
- 关系：他对绘宝的要求或相处方式

不要抽：客套寒暄、临时情绪、问句、一次性的查询、任何你的推测。
每行一条，格式： 类型|内容
内容 25 字以内，用「绘梨衣」当主语，写陈述句。
最多 3 条。没有值得记的就只输出 NO。

对话：
用户：{user}
助手：{assistant}
"""


@dataclass
class MemRememberTool(FunctionTool):
    store: Any = field(repr=False, default=None)
    name: str = "memory_remember"
    description: str = (
        "把关于绘梨衣的长期信息写进你自己的记忆本。"
        "当他说到自己的偏好、计划、近况、约定，或者要求你记住什么的时候用它。"
        "闲聊和一次性内容不要记。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要记住的内容，第三人称陈述句"},
                "kind": {
                    "type": "string",
                    "description": "类型：事实/偏好/计划/事件/关系，默认事实",
                },
            },
            "required": ["text"],
        }
    )

    async def run(self, event: AstrMessageEvent, text: str, kind: str = "事实") -> str:
        if not (text or "").strip():
            return "错误：text 不能为空。"
        item = self.store.add(text, kind=kind, source="manual", weight=2.0)
        if not item:
            return "没写进去。"
        return f"记住了（{item['kind']}｜{item['id']}）：{item['text']}"


@dataclass
class MemSearchTool(FunctionTool):
    store: Any = field(repr=False, default=None)
    name: str = "memory_search"
    description: str = (
        "在你自己的记忆本里检索关于绘梨衣的长期信息。"
        "不确定他以前说过什么的时候先查一遍，别瞎猜。"
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "关键词或问题"},
                "limit": {"type": "integer", "description": "返回条数，1-8，默认 5"},
            },
            "required": ["query"],
        }
    )

    async def run(self, event: AstrMessageEvent, query: str, limit: int = 5) -> str:
        if not (query or "").strip():
            return "错误：query 不能为空。"
        limit = max(1, min(int(limit or 5), 8))
        hits = self.store.search(query, limit=limit, min_score=0.25)
        if not hits:
            return "记忆本里没有相关记录。"
        lines = []
        for _score, item in hits:
            lines.append(f"[{item.get('kind')}] {item.get('text')} (id={item.get('id')})")
        return "\n".join(lines)


@dataclass
class MemForgetTool(FunctionTool):
    store: Any = field(repr=False, default=None)
    name: str = "memory_forget"
    description: str = "删掉记忆本里某条错误或过时的记录，按关键词或 id 删。"
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "关键词或 id"}},
            "required": ["key"],
        }
    )

    async def run(self, event: AstrMessageEvent, key: str) -> str:
        n = self.store.remove(key)
        return f"删了 {n} 条。" if n else "没找到匹配的记录。"


@register(
    DISPLAY_NAME,
    "若剑清",
    "记事本：从对话里自动抽取并注入长期记忆（关于绘梨衣的事实/偏好/计划）",
    "0.1.0",
)
class MemorizePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None) -> None:
        super().__init__(context)
        self.config = config or {}
        data_dir = os.path.join(get_astrbot_plugin_data_path(), PLUGIN_NAME)
        self.store = MemoryStore(os.path.join(data_dir, "memory.jsonl"))

        self.enabled = self._bool("enabled", True)
        self.inject_enabled = self._bool("inject_enabled", True)
        self.inject_limit = max(1, min(self._int("inject_limit", 6), 12))
        self.min_score = self._float("min_score", 0.3)
        self.recent_inject = max(0, min(self._int("recent_inject", 4), 10))
        self.auto_extract = self._bool("auto_extract", True)
        self.extract_provider = str(self.config.get("extract_provider", "") or "").strip()
        self.extract_gap = max(0, self._int("extract_gap", 20))

        self._last_extract: dict[str, float] = {}
        self._tasks: set[asyncio.Task] = set()

        self.context.add_llm_tools(
            MemRememberTool(store=self.store),
            MemSearchTool(store=self.store),
            MemForgetTool(store=self.store),
        )
        logger.info(
            "[Memorize] 记事本加载完成：%s 条记忆，路径 %s",
            self.store.count(),
            self.store.path,
        )

    # ---------------- 配置 ----------------

    def _bool(self, key: str, default: bool) -> bool:
        v = self.config.get(key, default)
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "on"}

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except Exception:
            return default

    def _float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except Exception:
            return default

    # ---------------- 注入 ----------------

    @filter.on_llm_request()
    async def inject_memory(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not (self.enabled and self.inject_enabled):
            return
        query = (req.prompt or event.message_str or "").strip()
        if not query:
            return
        hits = self.store.search(
            query,
            limit=self.inject_limit,
            min_score=self.min_score,
            recent=self.recent_inject,
        )
        if not hits:
            return
        lines = []
        for _score, item in hits:
            ts = time.strftime("%m-%d", time.localtime(float(item.get("ts") or 0)))
            lines.append(f"- [{item.get('kind')}|{ts}] {item.get('text')}")
            self.store.touch(str(item.get("id")))
        block = (
            "\n\n【你记得的事（你自己记的长期记忆，不是这轮对话里的内容）】\n"
            + "\n".join(lines)
            + "\n用法：跟当前话题相关的自然带出来，别整段复述、别当成需要汇报的清单；"
            "和这轮对话冲突时以这轮为准，冲突了就顺手用 memory_forget 删掉旧的。"
        )
        if req.system_prompt:
            req.system_prompt += block
        else:
            req.system_prompt = block.strip()

    # ---------------- 自动抽取 ----------------

    @filter.on_llm_response()
    async def schedule_extract(self, event: AstrMessageEvent, resp: Any) -> None:
        if not (self.enabled and self.auto_extract):
            return
        user_text = (event.message_str or "").strip()
        reply = (getattr(resp, "completion_text", "") or "").strip()
        if len(user_text) < 4 or not reply:
            return
        if user_text.startswith("/") or user_text.startswith("／"):
            return
        umo = event.unified_msg_origin
        now = time.time()
        if now - self._last_extract.get(umo, 0) < self.extract_gap:
            return
        self._last_extract[umo] = now

        task = asyncio.create_task(self._extract(user_text, reply, umo))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _extract(self, user_text: str, reply: str, umo: str) -> None:
        try:
            provider = None
            if self.extract_provider:
                provider = self.context.get_provider_by_id(self.extract_provider)
            if provider is None:
                provider = self.context.get_using_provider(umo=umo)
            if provider is None:
                return
            prompt = EXTRACT_PROMPT.format(user=user_text[:800], assistant=reply[:600])
            r = await provider.text_chat(
                prompt=prompt,
                session_id=None,
                contexts=[],
                system_prompt="只输出抽取结果，不要解释。",
            )
            raw = (getattr(r, "completion_text", "") or "").strip()
            self._apply(raw, user_text)
        except Exception as e:  # 抽取失败绝不能影响正常聊天
            logger.debug("[Memorize] 抽取失败：%s", e)

    def _apply(self, raw: str, user_text: str) -> None:
        if not raw or raw.strip().upper().startswith("NO"):
            return
        added = 0
        for line in raw.splitlines():
            line = line.strip().lstrip("-•*0123456789. ").strip()
            if not line or "|" not in line and "｜" not in line:
                continue
            kind, text = re.split(r"[|｜]", line, maxsplit=1)
            kind = kind.strip().strip("【】[]")
            text = text.strip().strip("【】[]")
            if not text or len(text) > 120:
                continue
            if text in user_text and len(text) < 4:
                continue
            if self.store.add(text, kind=kind if kind in KINDS else "事实", source="auto"):
                added += 1
            if added >= 3:
                break

    # ---------------- 人用命令 ----------------

    @filter.command("记忆")
    async def memory_cmd(self, event: AstrMessageEvent):
        """记忆 查看 / 记忆 搜 关键词 / 记忆 存 内容 / 记忆 删 关键词"""
        args = (event.message_str or "").strip().split(maxsplit=2)
        body = args[1] if len(args) > 1 else ""
        rest = args[2] if len(args) > 2 else ""

        if body in ("", "看", "列表", "ls"):
            stats = self.store.stats()
            items = sorted(
                self.store.all(), key=lambda it: float(it.get("lu") or 0), reverse=True
            )[:15]
            if not items:
                yield event.plain_result("记忆本是空的。")
                return
            lines = [f"共 {stats['total']} 条：{stats['kinds']}", "最近更新的："]
            for it in items:
                d = time.strftime("%m-%d", time.localtime(float(it.get("ts") or 0)))
                lines.append(f"{it['id']} [{it['kind']}|{d}] {it['text']}")
            lines.append("（记忆 搜 xx / 记忆 删 id或词）")
            yield event.plain_result("\n".join(lines))
            return

        if body in ("搜", "查"):
            if not rest.strip():
                yield event.plain_result("搜什么？")
                return
            hits = self.store.search(rest, limit=8, min_score=0.2)
            if not hits:
                yield event.plain_result("没有。" )
                return
            yield event.plain_result("\n".join(f"[{i['kind']}] {i['text']}" for _s, i in hits))
            return

        if body in ("存", "记", "add"):
            if not rest.strip():
                yield event.plain_result("存什么？")
                return
            item = self.store.add(rest, source="manual", weight=2.0)
            yield event.plain_result(f"记住了：{item['text']}" if item else "没写进去。")
            return

        if body in ("删", "忘", "del"):
            if not rest.strip():
                yield event.plain_result("删哪条？")
                return
            n = self.store.remove(rest)
            yield event.plain_result(f"删了 {n} 条。" if n else "没找到。")
            return

        yield event.plain_result("用法：记忆 / 记忆 搜 xx / 记忆 存 xx / 记忆 删 id")
