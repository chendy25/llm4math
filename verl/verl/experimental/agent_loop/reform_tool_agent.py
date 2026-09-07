"""v2-compatible tool agent: execute <tool_call> without injecting a tool catalog.

SFT v2 / eval never listed tools in the prompt. Stock ToolAgentLoop passes
OpenAI schemas into the chat template (the v3 failure mode). This subclass
keeps the Hermes parser and tool execution, but builds tokens from the
dataset messages only.

Generation stops on ``</tool_call>`` so the model waits for a splice, matching
``rollout_singleshot``.
"""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop


@register("reform_tool_agent")
class ReformToolAgentLoop(ToolAgentLoop):
    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        self._assert_mm_supported(bool(agent_data.image_data or agent_data.video_data or agent_data.audio_data))
        prompt_ids = await self.ct_build_initial_tokens(
            agent_data.messages,
            tools=None,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        stops = list(sampling_params.get("stop") or [])
        if "</tool_call>" not in stops:
            stops.append("</tool_call>")
        sampling_params = {**sampling_params, "stop": stops}
        return await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        # Do not re-inject tool schemas when splicing tool messages.
        agent_data._active_tool_schemas = None  # type: ignore[attr-defined]
        return await super()._handle_processing_tools_state(agent_data)
