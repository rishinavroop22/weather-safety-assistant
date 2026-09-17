"""FastAPI application: a thin HTTP layer over the existing LangGraph.

    Browser -> POST /api/chat -> run_turn(graph, session_id, message) -> ChatResponse

The session id is the LangGraph thread id, so conversation memory is the graph's
own checkpointer; the API keeps no state of its own.
"""

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.graph.state import CompiledStateGraph

from app.api.presenter import http_status_for, to_chat_response
from app.api.schemas import ChatRequest, ChatResponse, ErrorBody, ErrorResponse, FieldProblem, HealthResponse
from app.api.settings import ApiSettings
from app.graph import run_turn
from app.policy import PolicyStore

logger = logging.getLogger(__name__)

GraphFactory = Callable[[], tuple[CompiledStateGraph, PolicyStore]]


def production_graph_factory() -> tuple[CompiledStateGraph, PolicyStore]:
    """Real policy files, live Open-Meteo, LLM settings from the environment."""
    from app.runtime import DEFAULT_POLICY_DIR, build_production_graph

    policy_store = PolicyStore(DEFAULT_POLICY_DIR)
    return build_production_graph(policy_store=policy_store), policy_store


def create_app(
    *,
    graph: CompiledStateGraph | None = None,
    policy_store: PolicyStore | None = None,
    settings: ApiSettings | None = None,
    graph_factory: GraphFactory = production_graph_factory,
) -> FastAPI:
    """Build the API.

    Pass ``graph`` and ``policy_store`` to inject a test graph. Otherwise the graph is
    built at startup with ``graph_factory``; if that fails (e.g. missing LLM settings)
    the app still starts, ``/health`` works, and ``/api/chat`` answers 503.
    """
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.graph is None:
            try:
                app.state.graph, app.state.policy_store = graph_factory()
                logger.info("Graph ready")
            except Exception as exc:   # configuration problems must not crash the process or leak details
                logger.error("Graph could not be built: %s: %s", type(exc).__name__, exc)
        yield

    app = FastAPI(title="Weather Safety Assistant API", version="1.0.0", lifespan=lifespan)
    app.state.graph = graph
    app.state.policy_store = policy_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            FieldProblem(
                field=".".join(str(part) for part in error["loc"] if part != "body") or "body",
                problem=error["msg"].removeprefix("Value error, "),
            )
            for error in exc.errors()
        ]
        body = ErrorResponse(error=ErrorBody(code="invalid_request", message="The request is invalid.", fields=fields))
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        body = ErrorResponse(error=ErrorBody(code="internal_error", message="Something went wrong. Please try again."))
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        responses={
            422: {"model": ErrorResponse, "description": "invalid request"},
            500: {"model": ErrorResponse, "description": "unexpected server error"},
            503: {"description": "service problem: a ChatResponse with reason llm_unavailable / weather_error / "
                                 "location_error, or an ErrorResponse if the assistant is not configured"},
        },
    )
    def chat(request: ChatRequest) -> Any:
        graph, store = app.state.graph, app.state.policy_store
        if graph is None or store is None:
            body = ErrorResponse(error=ErrorBody(code="service_unavailable", message="The assistant is not available right now."))
            return JSONResponse(status_code=503, content=body.model_dump())

        session_id = request.session_id or uuid.uuid4().hex
        started = time.perf_counter()
        state = run_turn(graph, session_id, request.message.strip())
        response = to_chat_response(session_id, state, store.get().vocabulary)
        logger.info(
            "chat session=%s status=%s reason=%s %.0fms",
            session_id[:8], response.status, response.reason, (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(status_code=http_status_for(response), content=response.model_dump())

    if settings.frontend_dist is not None:
        # Production: serve the built frontend from the same origin (registered after the API routes).
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app
