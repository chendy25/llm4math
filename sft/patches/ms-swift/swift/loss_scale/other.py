# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import ConfigLossScale


class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'


class IgnoreToolResponseLossScale(ConfigLossScale):
    """Zero-weight the tool output that is spliced into an assistant turn.

    Datasets that keep a whole agentic trajectory inside one assistant message
    (ReForm-style `<think>/<round>` traces) carry the retrieval results inline.
    Those tokens are injected by the runtime at inference time, never generated,
    so training on them wastes the objective and teaches the model to fabricate
    tool output. Also keeps the empty-think patterns, since `--loss_scale`
    accepts only one plugin.
    """

    loss_scale_config = 'ignore_tool_response.json'
