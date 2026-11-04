"""Run a single Handbook question or an interactive CLI conversation."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from checkpoint_store import postgres_checkpointer
from handbook_graph import ConversationStateError, GraphSettings, aask, aresume, build_graph, thread_config
from handbook_retriever import HandbookRetriever
from inspect_graph import inspect_thread
from node_models import create_models, create_node_model
from reranking import DEFAULT_RERANK_MODEL, LocalReranker


def create_model():
    """Compatibility helper for shared-model callers; CLI uses create_models()."""
    return create_node_model("writer")


async def inspect_conversation(args):
    async with postgres_checkpointer(setup=False) as saver:
        result = await inspect_thread(saver, args.thread_id, history=args.state_history)
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"Сохранено: {args.output.resolve()}")
    else:
        print(content)
    return 0


async def run_conversation(args, model):
    client = (QdrantClient(path=str(args.local_path)) if args.local_path else
              QdrantClient(url=args.url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=120))
    try:
        async with postgres_checkpointer() as checkpointer:
            reranker = None if args.no_rerank else LocalReranker(
                model=args.rerank_model, threads=args.threads, batch_size=args.rerank_batch_size)
            retrieve = HandbookRetriever(client, alias=args.alias, threads=args.threads,
                                         top_k=args.top_k, candidates=args.candidates,
                                         retrieval_k=args.retrieval_k, rerank_limit=args.rerank_limit,
                                         reranker=reranker)
            graph = build_graph(model, retrieve, GraphSettings(max_sources=args.top_k,
                                max_context_chars=args.max_context_chars), checkpointer=checkpointer)
            print(f"thread_id: {args.thread_id}", file=sys.stderr)
            while True:
                if args.resume:
                    response = await aresume(graph, thread_id=args.thread_id)
                else:
                    try:
                        question = args.question if args.question is not None else await asyncio.to_thread(input, "Вы: ")
                    except EOFError:
                        break
                    if args.question is None and question.strip() == "/exit":
                        break
                    response = await aask(graph, question, thread_id=args.thread_id)
                print(json.dumps(response.model_dump(), ensure_ascii=False, indent=2) if args.json else response.answer)
                if args.question is not None or args.resume:
                    return 1 if response.reason in {"llm_error", "retrieval_error"} else 0
    finally:
        client.close()
    return 0


def main():
    # Load only this project's .env; never overwrite explicitly supplied env vars.
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Omit for an interactive chat; /exit to stop")
    parser.add_argument("--alias", default=os.getenv("QDRANT_ALIAS", "handbook"))
    parser.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--local-path", type=Path, help="Existing embedded Qdrant database")
    parser.add_argument("--json", action="store_true", help="Include status, sources, reason and node trace")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8, help="Final parent documents sent to the graph")
    parser.add_argument("--candidates", type=int, default=100, help="Child candidates per dense/BM25 branch")
    parser.add_argument("--retrieval-k", type=int, default=30, help="Parent candidates per research query")
    parser.add_argument("--rerank-limit", type=int, default=40, help="Maximum unique parents to rerank")
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--no-rerank", action="store_true", help="Disable local reranking for comparison")
    parser.add_argument("--thread-id", help="Reuse this ID to continue a saved conversation; default: new UUID")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume a pending run for --thread-id, without a new question")
    mode.add_argument("--state", action="store_true", help="Read the latest saved state; no model calls")
    mode.add_argument("--state-history", action="store_true", help="Read all saved states, oldest first; no model calls")
    parser.add_argument("--output", type=Path, help="Write --state / --state-history JSON to a UTF-8 file")
    args = parser.parse_args()
    inspection = args.state or args.state_history
    if (args.resume or inspection) and (not args.thread_id or args.question is not None):
        parser.error("--resume / --state / --state-history require --thread-id and no question")
    if args.output and not inspection:
        parser.error("--output requires --state or --state-history")
    if not inspection and (not 0 < args.top_k <= args.retrieval_k <= args.candidates
                           or args.rerank_limit < args.top_k or args.rerank_batch_size < 1
                           or args.max_context_chars < 1 or args.threads < 1):
        parser.error("Require candidates >= retrieval-k >= top-k > 0, rerank-limit >= top-k and positive batch/context/threads")
    args.thread_id = args.thread_id or str(uuid.uuid4())
    try:
        thread_config(args.thread_id)
        model = None if inspection else create_models()
    except ValueError as exc:
        parser.error(str(exc))
    try:
        # Psycopg async connections require a selector loop on Windows.
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
            return runner.run(inspect_conversation(args) if inspection else run_conversation(args, model))
    except KeyboardInterrupt:
        return 130
    except ConversationStateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        # Do not echo DSNs or provider credentials from connection error messages.
        print(f"Ошибка запуска/сохранения графа ({type(exc).__name__}). Проверьте PostgreSQL и CHECKPOINT_DATABASE_URL. "
              "Если запуск прерван, повторите с --resume --thread-id и тем же ID.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
