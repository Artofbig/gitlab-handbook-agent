"""Start the local API with a psycopg-compatible event loop on Windows."""
import argparse
import asyncio
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = uvicorn.Config("api:app", host=args.host, port=args.port, loop="asyncio", workers=1)
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None) as runner:
        runner.run(uvicorn.Server(config).serve())


if __name__ == "__main__":
    main()
