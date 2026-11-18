"""PathSnap API (Laptop 1) -- Pranav.

    uvicorn main:app --host 0.0.0.0 --port 8000

Every request runs the full pipeline dynamically: parse files -> clean -> candidates -> features -> classifier -> HMM/Viterbi.
Nothing is precomputed or hard-coded per trip; drop a new <id>/<id>.{track,nodes,arcs} folder into an evaluation directory
and it is immediately selectable.
"""
from __future__ import annotations
import os, re, threading, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathsnap.config import load_config, resolve, get_logger, set_seed
from pathsnap.data_io import discover_trips
from pathsnap.pipeline import PathSnapPipeline

log = get_logger("pathsnap.api")
cfg = load_config()
set_seed(cfg["seed"])
_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

app = FastAPI(title="PathSnap API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])   # UI runs on Laptop 2

_pipe: PathSnapPipeline | None = None
_lock = threading.Lock()                 # pipeline is CPU-bound; serialise to keep latency predictable
_cache: dict[str, dict] = {}


def data_roots() -> list[Path]:
    env = os.environ.get("PATHSNAP_DATA_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    root = resolve(cfg["paths"]["data_root"])
    return [root / d for d in cfg["paths"]["eval_dirs"]]


def pipeline() -> PathSnapPipeline:
    global _pipe
    if _pipe is None:
        _pipe = PathSnapPipeline(cfg)
        log.info("model=%s hmm=%s", _pipe.bundle["name"], _pipe.params.to_dict())
    return _pipe


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for ln in f if ln.strip())


class MatchRequest(BaseModel):
    trip_id: str
    include_classifier_route: bool = True


@app.on_event("startup")
def _startup():
    pipeline()                            # fail fast if artifacts are missing


@app.get("/health")
def health():
    p = pipeline()
    return {"status": "ok", "model": p.bundle["name"], "trips_available": len(discover_trips(data_roots()))}


@app.get("/trips")
def trips():
    out = []
    for tid, folder in discover_trips(data_roots()).items():
        out.append({"trip_id": tid, "n_points": _count_lines(folder / f"{tid}.track"),
                    "has_reference": (folder / f"{tid}.route").exists()})
    return {"trips": out}


def _match(trip_id: str, with_clf: bool) -> dict:
    if not _ID.match(trip_id):
        raise HTTPException(400, "invalid trip_id")
    folder = discover_trips(data_roots()).get(trip_id)
    if folder is None:
        raise HTTPException(404, f"trip {trip_id} not found in {[str(r) for r in data_roots()]}")
    key = f"{trip_id}:{with_clf}"
    if key in _cache:
        return {**_cache[key], "cached": True}
    with _lock:
        t0 = time.time()
        r = pipeline().run(folder, trip_id)
    r = {k: v for k, v in r.items() if not k.startswith("_")}
    if not with_clf:
        r["classifier_route"] = None
    r["model"] = pipeline().bundle["name"]
    r["cached"] = False
    _cache[key] = r
    log.info("matched %s: %d pts in %.2fs", trip_id, r["n_points"], time.time() - t0)
    return r


@app.post("/match")
def match(req: MatchRequest):
    return _match(req.trip_id, req.include_classifier_route)


@app.get("/match/{trip_id}")
def match_get(trip_id: str, classifier: bool = True):
    return _match(trip_id, classifier)


@app.delete("/cache")
def clear_cache():
    _cache.clear()
    return {"cleared": True}


@app.get("/")
def ui():
    return FileResponse(resolve("ui", "index.html"))
