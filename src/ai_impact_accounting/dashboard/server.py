"""FastAPI server for the DIA web dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .api import (
    GraphView,
    RowFilter,
    base_choices,
    clear_card_disclosure_cache,
    dashboard_payload,
    dataset_meta,
    export_csv,
    hub_ingest,
    hub_lookup,
)


STATIC_DIR = Path(__file__).parent / "static"
_HTML_NO_CACHE = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _parse_bool(val: str | None) -> bool:
    return (val or "").lower() in ("1", "true", "yes")


def _parse_row_filter(val: str | None) -> RowFilter:
    mapping = {
        "all": "all",
        "reporting": "reporting",
        "reporting only": "reporting",
        "nonzero": "nonzero",
        "carbon > 0 only": "nonzero",
    }
    return mapping.get((val or "all").lower(), "all")  # type: ignore[return-value]


def _parse_graph_view(val: str | None) -> GraphView:
    if (val or "").lower() in ("family", "selected family only"):
        return "family"
    return "all"


def create_app(
    store: Any,
    default_base: str = "meta-llama/Llama-3-8B",
    on_refresh: Optional[Callable[[], None]] = None,
) -> FastAPI:
    """Build the FastAPI app serving static UI + JSON API."""
    app = FastAPI(title="DIA — Data & Impact Accounting", docs_url="/api/docs", redoc_url=None)

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        return {**dataset_meta(store), "default_base": default_base}

    @app.get("/api/bases")
    async def bases() -> dict[str, Any]:
        choices = base_choices(store)
        if default_base and default_base not in choices:
            choices = [default_base, *choices]
        return {"bases": choices, "default_base": default_base}

    @app.get("/api/dashboard")
    async def dashboard(request: Request) -> JSONResponse:
        params = request.query_params
        base = params.get("base") or default_base
        payload = dashboard_payload(
            store,
            base=base,
            impute=_parse_bool(params.get("impute")),
            row_filter=_parse_row_filter(params.get("row_filter")),
            compare_base=params.get("compare") or "",
            graph_view=_parse_graph_view(params.get("graph_view")),
        )
        status = 200 if payload.get("ok") else 400
        return JSONResponse(payload, status_code=status)

    @app.post("/api/refresh")
    async def refresh() -> dict[str, Any]:
        store.load()
        clear_card_disclosure_cache()
        if on_refresh:
            on_refresh()
        return {"ok": True, **dataset_meta(store)}

    @app.get("/api/hub-lookup")
    async def hub_lookup_route(request: Request) -> JSONResponse:
        model = request.query_params.get("model") or default_base
        payload = hub_lookup(store, model)
        status = 200 if payload.get("ok") else 400
        return JSONResponse(payload, status_code=status)

    @app.post("/api/hub-ingest")
    async def hub_ingest_route(request: Request) -> JSONResponse:
        model = request.query_params.get("model") or default_base
        payload = hub_ingest(store, model)
        status = 200 if payload.get("ok") else 400
        return JSONResponse(payload, status_code=status)

    @app.get("/api/export.csv")
    async def csv_export(request: Request) -> Response:
        params = request.query_params
        base = params.get("base") or default_base
        text = export_csv(
            store,
            base=base,
            impute=_parse_bool(params.get("impute")),
            row_filter=_parse_row_filter(params.get("row_filter")),
        )
        filename = f"dia-footprint-{base.replace('/', '_')}.csv"
        return Response(
            content=text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Serve the shell HTML with no-cache so Space restarts / deploys are not
    # masked by a sticky browser or CDN copy of an older index.html (e.g. missing
    # nav links). CSS/JS keep their ?v= query busting.
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers=_HTML_NO_CACHE)

    @app.get("/index.html")
    async def index_html() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers=_HTML_NO_CACHE)

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def register_ingest_webhook(
    app: FastAPI,
    handler: Callable[..., Any],
    *,
    webhook_secret: str | None = None,
    path: str = "/webhooks/webhooks/ingest",
) -> None:
    """Register the HF ingest webhook on a FastAPI app (Hub path convention)."""
    from huggingface_hub._webhooks_server import _wrap_webhook_to_check_secret  # noqa: PLC0415
    from starlette.routing import Mount  # noqa: PLC0415

    route_handler = handler
    if webhook_secret:
        route_handler = _wrap_webhook_to_check_secret(handler, webhook_secret=webhook_secret)
    app.post(path)(route_handler)
    # ``create_app`` mounts StaticFiles at "/", which matches every path and would
    # shadow this POST route (405). Move the just-added route ahead of that mount.
    routes = app.router.routes
    new_route = routes.pop()
    insert_at = next((i for i, r in enumerate(routes) if isinstance(r, Mount)), len(routes))
    routes.insert(insert_at, new_route)


def serve(
    store: Any,
    *,
    host: str = "0.0.0.0",
    port: int = 7860,
    default_base: str = "meta-llama/Llama-3-8B",
    on_refresh: Optional[Callable[[], None]] = None,
) -> None:
    """Run the dashboard with uvicorn."""
    import uvicorn  # noqa: PLC0415

    app = create_app(store, default_base=default_base, on_refresh=on_refresh)
    uvicorn.run(app, host=host, port=port, log_level="info")
