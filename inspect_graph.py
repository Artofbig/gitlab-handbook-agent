"""Read persisted graph states without creating models or running any graph nodes."""
from handbook_graph import AgentState, AnswerDraft, ConversationStateError, citation_issues, thread_config


def checkpoint_view(item):
    checkpoint = item.checkpoint
    state = {key: value for key, value in checkpoint["channel_values"].items()
             if key in AgentState.__annotations__}
    diagnostics = {}
    if state.get("draft", {}).get("status") == "ok":
        draft = AnswerDraft.model_validate(state["draft"])
        diagnostics["citation_issues"] = citation_issues(draft, state.get("sources", []))
    return {
        "checkpoint_id": checkpoint["id"],
        "parent_checkpoint_id": (item.parent_config or {}).get("configurable", {}).get("checkpoint_id"),
        "created_at": checkpoint["ts"],
        "metadata": item.metadata,
        "updated_channels": checkpoint.get("updated_channels", []),
        "state": state,
        "diagnostics": diagnostics,
        "pending_writes": item.pending_writes or [],
    }


async def inspect_thread(saver, thread_id, *, history=False):
    config = thread_config(thread_id)
    if history:
        items = [checkpoint_view(item) async for item in saver.alist(config)]
        if not items:
            raise ConversationStateError("No checkpoint found for this thread_id")
        # Saver lists newest first. Present execution in chronological order.
        return {"thread_id": thread_id, "checkpoints": list(reversed(items))}
    item = await saver.aget_tuple(config)
    if item is None:
        raise ConversationStateError("No checkpoint found for this thread_id")
    return {"thread_id": thread_id, **checkpoint_view(item)}
