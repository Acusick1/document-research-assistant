from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from sse_starlette.sse import EventSourceResponse

from research_assistant.pipeline import AnswerEvent, RagPipeline, StatusEvent

logger = logging.getLogger(__name__)

pipeline: RagPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global pipeline  # noqa: PLW0603
    pipeline = RagPipeline()
    yield


app = FastAPI(title="Document Research Assistant", lifespan=lifespan)


@app.get("/query")
async def query(q: str = Query(description="Research question")) -> EventSourceResponse:
    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in pipeline.stream(q):
                match event:
                    case StatusEvent(message=msg):
                        yield {"event": "status", "data": msg}
                    case AnswerEvent(output=output):
                        yield {"event": "answer", "data": output.model_dump_json()}
        except Exception:
            logger.exception("Pipeline error for query: %s", q)
            yield {"event": "error", "data": json.dumps({"detail": "Internal pipeline error"})}

    return EventSourceResponse(event_generator())
