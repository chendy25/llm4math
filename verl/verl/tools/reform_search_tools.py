"""Native verl tools for ReForm retrieval: leansearch, physsearch, web_search.

Registered via ``examples/grpo_trainer/reform_tool_config.yaml``
(``multi_turn.tool_config_path``). Hermes parses the same ``<tool_call>``
JSON the v2 SFT model emits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_REFORM_TRAIN = Path(
    os.environ.get(
        "REFORM_TRAIN",
        "/mnt/afs_reason/chendongyang/code/llm4math/ReForm/data/train",
    )
)
if str(_REFORM_TRAIN) not in sys.path:
    sys.path.insert(0, str(_REFORM_TRAIN))

import leansearch_tool  # noqa: E402
import physsearch_tool  # noqa: E402
import web_search_tool  # noqa: E402

_SEARCH = os.environ.get("SEARCH_URL", "http://10.120.1.119:8000").rstrip("/")
leansearch_tool.DEFAULT_URL = _SEARCH
leansearch_tool.DEFAULT_CORPUS = "mathlib"
physsearch_tool.DEFAULT_URL = _SEARCH
if hasattr(physsearch_tool, "DEFAULT_CORPUS"):
    physsearch_tool.DEFAULT_CORPUS = "physlib"


def _wrap(body: str) -> str:
    text = (body or "").strip() or "(empty tool response)"
    return f"<tool_response>\n{text}\n</tool_response>"


class _ReformSearchTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return (instance_id or str(uuid4())), ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        raise NotImplementedError


class LeanSearchTool(_ReformSearchTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        query = str(parameters.get("query") or parameters.get("q") or "")
        num_results = int(parameters.get("num_results") or parameters.get("top_k") or 8)
        text = _wrap(
            leansearch_tool.run_tool_call(
                {"name": "leansearch", "arguments": {"query": query, "num_results": num_results}},
                url=_SEARCH,
                corpus="mathlib",
            )
        )
        return ToolResponse(text=text), 0.0, {}


class PhysSearchTool(_ReformSearchTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        query = str(parameters.get("query") or parameters.get("q") or "")
        num_results = int(parameters.get("num_results") or parameters.get("top_k") or 8)
        text = _wrap(
            physsearch_tool.run_tool_call(
                {"name": "physsearch", "arguments": {"query": query, "num_results": num_results}},
                url=_SEARCH,
            )
        )
        return ToolResponse(text=text), 0.0, {}


class WebSearchTool(_ReformSearchTool):
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        query = str(parameters.get("query") or parameters.get("q") or "")
        num_results = int(parameters.get("num_results") or parameters.get("top_k") or 5)
        text = _wrap(
            web_search_tool.run_tool_call(
                {"name": "web_search", "arguments": {"query": query, "num_results": num_results}},
            )
        )
        return ToolResponse(text=text), 0.0, {}
