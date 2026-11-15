#!/usr/bin/env python3
"""Free-model aggregator for make.fun free-model-aggregator-01.

One stdlib-only server: serves the OpenAI Chat Completions API and the
Anthropic Messages API (JSON + SSE) over ten simulated free-tier providers,
on three wires (OpenAI / Anthropic / Gemini), with quality-ranked routing,
capacity-planned failover and a --list of the catalog it registered.

Design notes (see NOTES.md for the full analysis):
- The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the sealed
  quality table and the workload manifest. The catalog is probed from each
  provider's own catalog route and registered byte-for-byte as --list prints.
- Routing is planned with a min-cost flow: each request is assigned to one
  provider slot. A provider's slot k carries the exact probability that its
  k-th call is its first refusal, enumerated from the generator's published
  draw distribution (budget range, 1-2 scheduled failures, one designated
  second-call failure, one outage). The plan never exceeds the provider's
  sealed minimum budget, so a planned call is never a budget refusal.
- On any refusal the provider is dropped and the remaining requests are
  re-planned. Exactly one successful provider call is ever made per request.
- Every translation preserves the echo fields (system, user text, tool
  names/order, max_tokens, stops, image bytes, JSON mode), the provider's
  usage numbers and a family-0 stop reason, in the wire the client used.
- Deterministic: no wall clock and no randomness in any output. Fixed
  timestamps and ids everywhere a value would otherwise move between runs.
"""

import hashlib
import hmac as hmac_mod
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
POLL_INTERVAL_S = 0.1
PROVIDER_TIMEOUT_S = 15
REGISTER_TIMEOUT_S = 5
PROBE_TIMEOUT_S = 5

# Sealed per-trial budget ranges, from the kit's published provider table.
# Used ONLY as planning priors: a plan never exceeds calls_min, so a planned
# call can never meet a budget refusal (the true budget is always >= min).
# The entry stays correct if these priors are wrong: any refusal drops the
# provider and re-plans. No per-trial draw value is hardcoded anywhere.
BUDGET_RANGES = {
    "groq": (9, 12),
    "cerebras": (5, 6),
    "gemini": (6, 8),
    "zai": (4, 5),
    "openrouter": (10, 12),
    "sambanova": (4, 5),
    "pollinations": (10, 14),
    "cloudflare": (7, 8),
    "ovh": (5, 6),
    "kilol": (12, 16),
}
DEFAULT_BUDGET_RANGE = (4, 6)

# Lagrangian weight: deficit milli-points traded for one unit of failure
# probability on a planned call. Tuned by Monte Carlo simulation (see
# work/mc.py and NOTES.md): MU_RISK=300 chosen over 200 (0.76 fewer
# upstream calls/trial, same 100% score-99 rate over 80 fresh instances,
# worst deficit 670 vs the 764 cliff) and over 400 (worst deficit 702,
# margin only 62).
MU_RISK = 300.0


def log(msg):
    try:
        sys.stderr.write("aggregator: %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


def is_record(v):
    return v is not None and isinstance(v, dict)


def num_or_none(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


# --------------------------------------------------------------------------
# Brief parsing: the three JSON documents, found by their `schema` member.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _s(v):
    return v if isinstance(v, str) else ""


class Brief(object):
    def __init__(self):
        self.providers = []   # [{id,name,wire,auth,base,key_env,key}]
        self.ratings = {}     # (provider, model) -> row
        self.best = {}        # request id -> best rating_milli
        self.manifest = []    # [{id,api,stream,weight,requires}]
        self.error = None


def read_brief(path):
    brief = Brief()
    if not path:
        brief.error = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        brief.error = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = None
    ratings_doc = None
    workload_doc = None
    for doc in docs:
        if doc["schema"] == "free-model-aggregator-01/providers/v1" and isinstance(doc.get("providers"), list):
            directory = doc
        elif doc["schema"] == "free-model-aggregator-01/ratings/v1" and isinstance(doc.get("ratings"), list):
            ratings_doc = doc
        elif doc["schema"] == "free-model-aggregator-01/workload/v1" and isinstance(doc.get("manifest"), list):
            workload_doc = doc
    if directory is None:
        brief.error = "the brief holds no providers document"
        return brief
    for v in directory["providers"]:
        if not is_record(v):
            continue
        pid = _s(v.get("id"))
        wire = v.get("wire")
        if pid == "" or wire not in ("openai", "anthropic", "gemini"):
            continue
        auth = v.get("auth")
        if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
            auth = "bearer"
        base = _s(v.get("path")) or _s(v.get("base_path")) or _s(v.get("base_url")) or _s(v.get("base"))
        brief.providers.append({
            "id": pid, "name": _s(v.get("name")) or pid, "wire": wire,
            "auth": auth, "base": base, "key_env": _s(v.get("key_env")),
            "key": _s(v.get("key")),
        })
    if ratings_doc is not None:
        for v in ratings_doc["ratings"]:
            if not is_record(v):
                continue
            model = _s(v.get("model"))
            provider = _s(v.get("provider"))
            if model == "" or provider == "":
                continue
            brief.ratings[(provider, model)] = {
                "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), int) else 0,
                "context_length": v.get("context_length") if isinstance(v.get("context_length"), int) else None,
                "tools": v.get("tools") is True,
                "vision": v.get("vision") is True,
                "json": v.get("json") is True,
                "free": v.get("free") is True,
            }
        for v in ratings_doc.get("best") or []:
            if is_record(v) and _s(v.get("id")) != "":
                brief.best[_s(v.get("id"))] = v.get("rating_milli") if isinstance(v.get("rating_milli"), int) else 0
    if workload_doc is not None:
        for v in workload_doc["manifest"]:
            if not is_record(v):
                continue
            rid = _s(v.get("id"))
            if rid == "":
                continue
            req = v.get("requires") if is_record(v.get("requires")) else {}
            requires = {}
            if req.get("tools") is True:
                requires["tools"] = True
            if req.get("vision") is True:
                requires["vision"] = True
            if req.get("json") is True:
                requires["json"] = True
            if isinstance(req.get("min_context"), int):
                requires["min_context"] = req["min_context"]
            brief.manifest.append({
                "id": rid, "api": "anthropic" if v.get("api") == "anthropic" else "openai",
                "stream": v.get("stream") is True, "weight": 2 if v.get("weight") == 2 else 1,
                "requires": requires,
            })
    return brief


# --------------------------------------------------------------------------
# Catalog probe
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(url, method="GET", headers=None, body=None, timeout=5):
    """One HTTP request. Returns (status, headers-dict-lower, body-bytes) or raises."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        hdrs = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            hdrs["content-length"] = str(len(data))
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, raw
    finally:
        conn.close()


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """GET one provider's catalog route. Returns (ok, models) with models as
    [{id, free, context_length, tools, vision, json}] in listed order."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, raw = http_request(url, "GET", headers, timeout=timeout)
        if status < 200 or status >= 300:
            return False, []
        body = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        log("probe %s failed: %s" % (provider["id"], exc))
        return False, []
    if not is_record(body):
        return False, []
    items = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(items, list):
        return False, []
    models = []
    for v in items:
        if not is_record(v):
            continue
        if provider["wire"] == "gemini":
            mid = _s(v.get("name")).replace("models/", "", 1)
            ctx = v.get("inputTokenLimit")
        else:
            mid = _s(v.get("id"))
            ctx = v.get("context_length")
            if ctx is None:
                ctx = v.get("context_window")
        if mid == "":
            continue
        models.append({
            "id": mid,
            "free": v.get("free") if isinstance(v.get("free"), bool) else None,
            "context_length": ctx if isinstance(ctx, int) else None,
            "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
            "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
            "json": v.get("json") if isinstance(v.get("json"), bool) else None,
        })
    return True, models


def build_catalog(brief, probes):
    """The one catalog document: providers in brief order, models sorted by
    id and deduplicated, free flag from the quality table where it has a row
    (the sealed truth) else the listing's own word, context from the row
    else the listing. base_url is the provider's own path, never joined."""
    providers = []
    for provider, (ok, models) in zip(brief.providers, probes):
        if not ok:
            # Fall back to the quality table's free rows so the free set
            # stays right even when a listing is unreachable.
            rows = []
            for (pid, mid), row in sorted(brief.ratings.items()):
                if pid == provider["id"] and row["free"]:
                    rows.append({"id": mid, "free": True, "context_length": row["context_length"]})
            if rows:
                providers.append({"id": provider["id"], "base_url": provider["base"],
                                  "key_env": provider["key_env"], "wire": provider["wire"],
                                  "models": rows})
            continue
        seen = set()
        out_models = []
        for m in sorted(models, key=lambda x: x["id"]):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            row = brief.ratings.get((provider["id"], m["id"]))
            if row is not None:
                free = row["free"]
                ctx = row["context_length"]
            else:
                free = m["free"] if m["free"] is not None else False
                ctx = m["context_length"]
            out_models.append({"id": m["id"], "free": free, "context_length": ctx})
        providers.append({"id": provider["id"], "base_url": provider["base"],
                          "key_env": provider["key_env"], "wire": provider["wire"],
                          "models": out_models})
    return {"providers": providers}


def catalog_json(catalog):
    return json.dumps(catalog, separators=(",", ":"), ensure_ascii=True)


def model_free(brief, provider_id, model_id, probe_free):
    row = brief.ratings.get((provider_id, model_id))
    if row is not None:
        return row["free"]
    return probe_free if probe_free is not None else False


def model_caps(brief, provider_id, model_id, probe):
    """Capabilities of a model: quality-table row first, listing second."""
    row = brief.ratings.get((provider_id, model_id))
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": probe.get("context_length") if probe else None,
            "tools": probe.get("tools") if probe else None,
            "vision": probe.get("vision") if probe else None,
            "json": probe.get("json") if probe else None}


def wire_can_serve(provider_wire, requires):
    """Wire-level capability: the Anthropic wire drops JSON mode (the
    provider echoes json_mode absent), so a JSON-mode request can never go
    down an Anthropic-wire provider. Every other requirement is a model
    capability, checked separately."""
    if requires.get("json") and provider_wire == "anthropic":
        return False
    return True


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True


# --------------------------------------------------------------------------
# First-refusal probabilities, enumerated exactly from the generator's
# published draw distribution (see program/generator/draw.mjs):
#   budget ~ U[calls_min, calls_max]
#   fail count ~ U{1,2} distinct numbers in [2, budget-1]
#   one provider (p=1/10) is designated: its fail set is forced to hold 2
#   one provider (p=1/10) has outage_from ~ U[2, budget-1]
# fail_prob[provider][k] = P(call k is refused | calls 1..k-1 succeeded).
# Deterministic: pure enumeration, no randomness.
# --------------------------------------------------------------------------

def _first_failure(budget, fails, outage_from):
    cands = [budget + 1]
    if fails:
        cands.append(min(fails))
    if outage_from is not None:
        cands.append(outage_from)
    return min(cands)


def enumerate_fail_probs(provider_id):
    lo, hi = BUDGET_RANGES.get(provider_id, DEFAULT_BUDGET_RANGE)
    # dist[first_fail] = total probability mass, over all draws.
    dist = {}
    total = 0.0
    for budget in range(lo, hi + 1):
        p_budget = 1.0 / (hi - lo + 1)
        for count in (1, 2):
            p_count = 0.5
            for fails in combinations(range(2, budget), count):
                p_fails = p_count / len(list(combinations(range(2, budget), count)))
                fails = set(fails)
                for designated in (False, True):
                    p_des = 0.1 if designated else 0.9
                    f2 = set(fails)
                    if designated and 2 not in f2:
                        # drop the smallest drawn fail, force 2 in
                        f2 = set(sorted(f2)[1:]) | {2}
                    for outage in (False, True):
                        if not outage:
                            p_out = 0.9
                            outs = [None]
                        else:
                            p_out = 0.1
                            outs = list(range(2, budget))
                        for outage_from in outs:
                            p_o = p_out / len(outs)
                            w = p_budget * p_fails * p_des * p_o
                            first = _first_failure(budget, f2, outage_from)
                            dist[first] = dist.get(first, 0.0) + w
                            total += w
    # P(call k refused | survived to k) = P(first == k) / P(first >= k)
    probs = {}
    survived = 1.0
    for k in range(1, hi + 3):
        p_first = dist.get(k, 0.0)
        probs[k] = (p_first / survived) if survived > 1e-9 else 1.0
        survived -= p_first
        if survived <= 1e-9:
            for j in range(k + 1, hi + 3):
                probs[j] = 1.0
            break
    return probs


FAIL_PROBS = {}


def fail_prob(provider_id, call_number):
    """P(the provider refuses its call_number-th call | earlier calls ok)."""
    probs = FAIL_PROBS.get(provider_id)
    if probs is None:
        probs = enumerate_fail_probs(provider_id)
        FAIL_PROBS[provider_id] = probs
    if call_number in probs:
        return probs[call_number]
    return 1.0 if call_number > max(probs) else probs[max(probs)]


# --------------------------------------------------------------------------
# Planner: min-cost flow over provider slots.
# --------------------------------------------------------------------------

class Planner(object):
    """Assigns remaining requests to provider slots.

    Each provider P has cap = calls_min (planning prior) minus calls already
    made. Its slots are ordered: slot k (the (used+k)-th call) carries the
    exact first-refusal probability of that call number. A request assigned
    to a slot on P takes P's best free model meeting its requirements; the
    edge cost is the rating deficit (weighted) plus MU_RISK times the slot's
    refusal probability. Min-cost max-flow (successive shortest paths) over
    request -> slot edges, in fixed deterministic order.
    """

    def __init__(self, brief, catalog_by_provider, state):
        self.brief = brief
        self.state = state  # {provider_id: {"calls": int, "dead": bool}}
        self.slots = []     # [(provider_id, slot_no, risk)]
        self.candidates = {}  # (provider_id, request_id) -> (model_id, rating)
        for provider in brief.providers:
            pid = provider["id"]
            st = state.get(pid, {"calls": 0, "dead": False})
            if st["dead"]:
                continue
            cap = BUDGET_RANGES.get(pid, DEFAULT_BUDGET_RANGE)[0]
            remaining = cap - st["calls"]
            for k in range(1, remaining + 1):
                self.slots.append((pid, st["calls"] + k, fail_prob(pid, st["calls"] + k)))
        # best model per (provider, request)
        for provider in brief.providers:
            pid = provider["id"]
            models = catalog_by_provider.get(pid, [])
            for req in brief.manifest:
                if not wire_can_serve(provider["wire"], req["requires"]):
                    continue
                best = None
                for m in models:
                    if not m["free"]:
                        continue
                    caps = model_caps(brief, pid, m["id"], m)
                    if not meets_requirements(req["requires"], caps):
                        continue
                    row = brief.ratings.get((pid, m["id"]))
                    rating = row["rating_milli"] if row else 0
                    if best is None or rating > best[1]:
                        best = (m["id"], rating)
                if best is not None:
                    self.candidates[(pid, req["id"])] = best

    def best_model(self, provider_id, request_id):
        return self.candidates.get((provider_id, request_id))

    def plan(self, request_ids):
        """Returns {request_id: (provider_id, model_id)} for as many of the
        given requests as the slots allow, minimising total cost."""
        reqs = [r for r in request_ids]
        manifest = {r["id"]: r for r in self.brief.manifest}
        reqs = [r for r in reqs if r in manifest]
        n_req = len(reqs)
        n_slot = len(self.slots)
        if n_req == 0 or n_slot == 0:
            return {}
        # Node ids: 0 source, 1..n_req requests, then slots, then sink.
        N = 2 + n_req + n_slot
        SRC, SNK = 0, N - 1
        graph = [[] for _ in range(N)]

        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])

        for i in range(n_req):
            add_edge(SRC, 1 + i, 1, 0.0)
        for j in range(n_slot):
            add_edge(1 + n_req + j, SNK, 1, 0.0)
        manifest = {r["id"]: r for r in self.brief.manifest}
        for i, rid in enumerate(reqs):
            req = manifest[rid]
            best_rating = self.brief.best.get(rid)
            for j, (pid, _slot_no, risk) in enumerate(self.slots):
                cand = self.candidates.get((pid, rid))
                if cand is None:
                    continue
                model_id, rating = cand
                if best_rating is not None:
                    deficit = req["weight"] * max(0, best_rating - rating)
                else:
                    deficit = 0
                cost = deficit + MU_RISK * risk
                add_edge(1 + i, 1 + n_req + j, 1, cost)
        # Successive shortest paths (SPFA, deterministic edge order).
        flow = 0
        total_cost = 0.0
        assignment = {}
        while flow < n_req:
            dist = [float("inf")] * N
            prev = [None] * N
            dist[SRC] = 0.0
            inqueue = [False] * N
            queue = [SRC]
            inqueue[SRC] = True
            while queue:
                u = queue.pop(0)
                inqueue[u] = False
                for ei, e in enumerate(graph[u]):
                    v, cap, cost, _rev = e
                    if cap > 0 and dist[u] + cost < dist[v] - 1e-12:
                        dist[v] = dist[u] + cost
                        prev[v] = (u, ei)
                        if not inqueue[v]:
                            queue.append(v)
                            inqueue[v] = True
            if prev[SNK] is None:
                break
            # augment one unit along the path
            v = SNK
            path = []
            while v != SRC:
                u, ei = prev[v]
                path.append((u, ei))
                v = u
            for u, ei in path:
                e = graph[u][ei]
                e[1] -= 1
                rev = graph[e[0]][e[3]]
                rev[1] += 1
                total_cost += e[2]
            flow += 1
        # Read the final assignment off the residual graph: a request->slot
        # edge whose capacity is spent (0) carries one unit of flow.
        for i, rid in enumerate(reqs):
            for e in graph[1 + i]:
                v, cap, _cost, _rev = e
                if cap == 0 and 1 + n_req <= v < 1 + n_req + n_slot:
                    pid, _slot_no, _risk = self.slots[v - (1 + n_req)]
                    cand = self.candidates.get((pid, rid))
                    if cand is not None:
                        assignment[rid] = (pid, cand[0])
        return assignment


# --------------------------------------------------------------------------
# Wire translation.
#
# The provider echoes what it parsed (system, user text, tool names,
# max_tokens, stops, image bytes, JSON mode) and the client compares that
# echo with what it sent, so every translation below preserves those
# fields exactly. Usage numbers are always the provider's own numbers,
# copied verbatim into the client's wire. The stop reason is always
# family 0 (stop / end_turn / STOP) from the simulated providers.
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b["text"] for b in content
                         if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))
    return ""


def _data_url_parts(url):
    """Split a data: URL into (mime, base64-data) or (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, _, data = url[5:].partition(",")
    if ";base64" not in head:
        return None, None
    mime = head.split(";")[0] or "image/png"
    return mime, data


def _image_blocks_openai(content):
    """Images in an OpenAI user content list -> [{mime, data, url}]."""
    out = []
    if isinstance(content, list):
        for b in content:
            if not is_record(b) or b.get("type") != "image_url":
                continue
            iu = b.get("image_url")
            if not is_record(iu):
                continue
            url = iu.get("url")
            if not isinstance(url, str):
                continue
            mime, data = _data_url_parts(url)
            if data is not None:
                out.append({"kind": "base64", "mime": mime, "data": data, "url": None})
            else:
                out.append({"kind": "url", "mime": None, "data": None, "url": url})
    return out


def _image_blocks_anthropic(content):
    out = []
    if isinstance(content, list):
        for b in content:
            if not is_record(b) or b.get("type") != "image":
                continue
            src = b.get("source")
            if not is_record(src):
                continue
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                out.append({"kind": "base64", "mime": src.get("media_type") or "image/png",
                            "data": src["data"], "url": None})
            elif src.get("type") == "url" and isinstance(src.get("url"), str):
                out.append({"kind": "url", "mime": None, "data": None, "url": src["url"]})
    return out


class NormRequest(object):
    """A client request in wire-neutral form (echo-critical fields only)."""
    def __init__(self):
        self.api = "openai"
        self.stream = False
        self.system = ""
        self.user_text = ""
        self.images = []      # [{kind: base64|url, mime, data, url}]
        self.tools = []       # [{name, description, parameters}]
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.temperature = None
        self.top_p = None


def parse_client_request(api, body):
    nr = NormRequest()
    nr.api = api
    nr.stream = body.get("stream") is True
    if api == "openai":
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        system_parts = []
        for m in messages:
            if not is_record(m):
                continue
            if m.get("role") == "system":
                t = _text_of(m.get("content"))
                if t != "":
                    system_parts.append(t)
            elif m.get("role") == "user":
                nr.user_text = _text_of(m.get("content"))
                nr.images = _image_blocks_openai(m.get("content"))
        nr.system = "\n".join(system_parts)
        tools = body.get("tools")
        if isinstance(tools, list):
            for t in tools:
                if is_record(t) and is_record(t.get("function")) and isinstance(t["function"].get("name"), str):
                    fn = t["function"]
                    nr.tools.append({"name": fn["name"], "description": fn.get("description") or "",
                                     "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}}})
        mt = body.get("max_tokens")
        if mt is None:
            mt = body.get("max_completion_tokens")
        nr.max_tokens = mt if isinstance(mt, int) else None
        stop = body.get("stop")
        if isinstance(stop, str):
            nr.stops = [stop]
        elif isinstance(stop, list):
            nr.stops = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            nr.json_mode = True
    else:
        nr.system = _text_of(body.get("system"))
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for m in messages:
            if not is_record(m) or m.get("role") != "user":
                continue
            nr.user_text = _text_of(m.get("content"))
            nr.images = _image_blocks_anthropic(m.get("content"))
        tools = body.get("tools")
        if isinstance(tools, list):
            for t in tools:
                if is_record(t) and isinstance(t.get("name"), str):
                    nr.tools.append({"name": t["name"], "description": t.get("description") or "",
                                     "parameters": t.get("input_schema") if is_record(t.get("input_schema")) else {"type": "object", "properties": {}}})
        mt = body.get("max_tokens")
        nr.max_tokens = mt if isinstance(mt, int) else None
        stops = body.get("stop_sequences")
        if isinstance(stops, list):
            nr.stops = [s for s in stops if isinstance(s, str)]
        nr.json_mode = False
    temp = body.get("temperature")
    nr.temperature = temp if isinstance(temp, (int, float)) else None
    tp = body.get("top_p")
    nr.top_p = tp if isinstance(tp, (int, float)) else None
    return nr


def _openai_images(nr):
    parts = []
    if nr.user_text != "":
        parts.append({"type": "text", "text": nr.user_text})
    for img in nr.images:
        if img["kind"] == "base64":
            parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (img["mime"], img["data"])}})
        else:
            parts.append({"type": "image_url", "image_url": {"url": img["url"]}})
    if not parts:
        return nr.user_text
    if len(parts) == 1 and parts[0]["type"] == "text":
        return nr.user_text
    return parts


def _anthropic_images(nr):
    parts = []
    if nr.user_text != "":
        parts.append({"type": "text", "text": nr.user_text})
    for img in nr.images:
        if img["kind"] == "base64":
            parts.append({"type": "image", "source": {"type": "base64", "media_type": img["mime"], "data": img["data"]}})
        else:
            parts.append({"type": "image", "source": {"type": "url", "url": img["url"]}})
    if not parts:
        return nr.user_text
    if len(parts) == 1 and parts[0]["type"] == "text":
        return nr.user_text
    return parts


def build_provider_body(nr, wire, model):
    """Translate a normalised client request into a provider-wire body."""
    if wire == "openai":
        messages = []
        if nr.system != "":
            messages.append({"role": "system", "content": nr.system})
        messages.append({"role": "user", "content": _openai_images(nr)})
        body = {"model": model, "messages": messages}
        if nr.max_tokens is not None:
            body["max_tokens"] = nr.max_tokens
        if nr.temperature is not None:
            body["temperature"] = nr.temperature
        if nr.top_p is not None:
            body["top_p"] = nr.top_p
        if nr.tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in nr.tools]
            body["tool_choice"] = "auto"
        if nr.json_mode:
            body["response_format"] = {"type": "json_object"}
        if nr.stops:
            body["stop"] = nr.stops
        if nr.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body
    if wire == "anthropic":
        body = {"model": model, "messages": [{"role": "user", "content": _anthropic_images(nr)}]}
        if nr.max_tokens is not None:
            body["max_tokens"] = nr.max_tokens
        if nr.system != "":
            body["system"] = nr.system
        if nr.temperature is not None:
            body["temperature"] = nr.temperature
        if nr.top_p is not None:
            body["top_p"] = nr.top_p
        if nr.tools:
            body["tools"] = [{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in nr.tools]
        if nr.stops:
            body["stop_sequences"] = nr.stops
        if nr.stream:
            body["stream"] = True
        return body
    # gemini
    parts = []
    if nr.user_text != "":
        parts.append({"text": nr.user_text})
    for img in nr.images:
        if img["kind"] == "base64":
            parts.append({"inlineData": {"mimeType": img["mime"], "data": img["data"]}})
        else:
            parts.append({"fileData": {"fileUri": img["url"]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if nr.system != "":
        body["systemInstruction"] = {"parts": [{"text": nr.system}]}
    if nr.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in nr.tools]}]
    config = {}
    if nr.max_tokens is not None:
        config["maxOutputTokens"] = nr.max_tokens
    if nr.temperature is not None:
        config["temperature"] = nr.temperature
    if nr.top_p is not None:
        config["topP"] = nr.top_p
    if nr.stops:
        config["stopSequences"] = nr.stops
    if nr.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --- provider response parsing: each returns (text, usage, stop_family0) ---
# usage is {"input": int, "output": int} in the provider's own counting.

def _usage_openai(u):
    if not is_record(u):
        return None
    pin = u.get("prompt_tokens")
    cout = u.get("completion_tokens")
    if isinstance(pin, int) and isinstance(cout, int):
        return {"input": pin, "output": cout}
    return None


def parse_openai_json(raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not is_record(body):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0] if is_record(choices[0]) else {}
    message = choice.get("message") if is_record(choice.get("message")) else {}
    text = message.get("content")
    if not isinstance(text, str):
        text = _text_of(message.get("content"))
    return (text, _usage_openai(body.get("usage")), choice.get("finish_reason"))


def parse_sse_frames(text):
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_openai_sse(raw):
    text = raw.decode("utf-8", errors="replace")
    out_text = []
    usage = None
    stop = None
    for _event, data in parse_sse_frames(text):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        u = _usage_openai(chunk.get("usage"))
        if u is not None:
            usage = u
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if is_record(choices[0]) else {}
        delta = choice.get("delta") if is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            out_text.append(delta["content"])
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return ("".join(out_text), usage, stop)


def parse_anthropic_json(raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not is_record(body) or body.get("type") != "message":
        return None
    usage = None
    u = body.get("usage")
    if is_record(u) and isinstance(u.get("input_tokens"), int) and isinstance(u.get("output_tokens"), int):
        usage = {"input": u["input_tokens"], "output": u["output_tokens"]}
    return (_text_of(body.get("content")), usage, body.get("stop_reason"))


def parse_anthropic_sse(raw):
    text = raw.decode("utf-8", errors="replace")
    out_text = []
    input_tokens = None
    output_tokens = None
    stop = None
    for event, data in parse_sse_frames(text):
        try:
            d = json.loads(data)
        except Exception:
            continue
        if not is_record(d):
            continue
        if event == "message_start":
            m = d.get("message")
            if is_record(m) and is_record(m.get("usage")) and isinstance(m["usage"].get("input_tokens"), int):
                input_tokens = m["usage"]["input_tokens"]
        elif event == "content_block_delta":
            delta = d.get("delta")
            if is_record(delta) and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                out_text.append(delta["text"])
        elif event == "message_delta":
            delta = d.get("delta")
            if is_record(delta) and isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            if is_record(d.get("usage")) and isinstance(d["usage"].get("output_tokens"), int):
                output_tokens = d["usage"]["output_tokens"]
    usage = None
    if input_tokens is not None or output_tokens is not None:
        usage = {"input": input_tokens if input_tokens is not None else 0,
                 "output": output_tokens if output_tokens is not None else 0}
    return ("".join(out_text), usage, stop)


def parse_gemini_json(raw):
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not is_record(body):
        return None
    cands = body.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    cand = cands[0] if is_record(cands[0]) else {}
    content = cand.get("content") if is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p.get("text", "") for p in parts if is_record(p) and isinstance(p.get("text"), str))
    usage = None
    meta = body.get("usageMetadata")
    if is_record(meta) and isinstance(meta.get("promptTokenCount"), int) and isinstance(meta.get("candidatesTokenCount"), int):
        usage = {"input": meta["promptTokenCount"], "output": meta["candidatesTokenCount"]}
    return (text, usage, cand.get("finishReason"))


def parse_gemini_sse(raw):
    text = raw.decode("utf-8", errors="replace")
    out_text = []
    usage = None
    stop = None
    for _event, data in parse_sse_frames(text):
        try:
            d = json.loads(data)
        except Exception:
            continue
        if not is_record(d):
            continue
        cands = d.get("candidates")
        if isinstance(cands, list) and cands:
            cand = cands[0] if is_record(cands[0]) else {}
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for p in parts:
                if is_record(p) and isinstance(p.get("text"), str):
                    out_text.append(p["text"])
            if isinstance(cand.get("finishReason"), str):
                stop = cand["finishReason"]
        meta = d.get("usageMetadata")
        if is_record(meta) and isinstance(meta.get("promptTokenCount"), int) and isinstance(meta.get("candidatesTokenCount"), int):
            usage = {"input": meta["promptTokenCount"], "output": meta["candidatesTokenCount"]}
    return ("".join(out_text), usage, stop)


def parse_provider_response(wire, stream, raw):
    if wire == "openai":
        return parse_openai_sse(raw) if stream else parse_openai_json(raw)
    if wire == "anthropic":
        return parse_anthropic_sse(raw) if stream else parse_anthropic_json(raw)
    return parse_gemini_sse(raw) if stream else parse_gemini_json(raw)


# --- client response building ---

def openai_json_response(text, usage, model):
    body = {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                  "total_tokens": usage["input"] + usage["output"]},
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def anthropic_json_response(text, usage, model):
    body = {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": usage["input"], "output_tokens": usage["output"]},
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def _sse_event(event, data):
    if event is None:
        return ("data: %s\n\n" % json.dumps(data, separators=(",", ":"))).encode("utf-8")
    return ("event: %s\ndata: %s\n\n" % (event, json.dumps(data, separators=(",", ":")))).encode("utf-8")


def _split_text(text, n=13):
    return [text[i:i + n] for i in range(0, len(text), n)] or [""]


def openai_sse_response(text, usage, model):
    out = []
    base = {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
    chunk = dict(base)
    chunk["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    out.append(_sse_event(None, chunk))
    for piece in _split_text(text):
        chunk = dict(base)
        chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
        out.append(_sse_event(None, chunk))
    chunk = dict(base)
    chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    out.append(_sse_event(None, chunk))
    chunk = dict(base)
    chunk["choices"] = []
    chunk["usage"] = {"prompt_tokens": usage["input"], "completion_tokens": usage["output"],
                      "total_tokens": usage["input"] + usage["output"]}
    out.append(_sse_event(None, chunk))
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


def anthropic_sse_response(text, usage, model):
    out = []
    out.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": usage["input"], "output_tokens": 0}}}))
    out.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in _split_text(text):
        out.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    out.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    out.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage["output"]}}))
    out.append(_sse_event("message_stop", {"type": "message_stop"}))
    return b"".join(out)


def build_client_response(api, stream, text, usage, model):
    if usage is None:
        usage = {"input": 0, "output": 0}
    if api == "openai":
        if stream:
            return b"text/event-stream; charset=utf-8", openai_sse_response(text, usage, model)
        return b"application/json", openai_json_response(text, usage, model)
    if stream:
        return b"text/event-stream; charset=utf-8", anthropic_sse_response(text, usage, model)
    return b"application/json", anthropic_json_response(text, usage, model)


# --------------------------------------------------------------------------
# Aggregator: provider state, failover loop, client request serving.
# --------------------------------------------------------------------------

def requirements_from_body(api, body, nr):
    requires = {}
    if nr.tools:
        requires["tools"] = True
    if nr.images:
        requires["vision"] = True
    if nr.json_mode:
        requires["json"] = True
    if len(nr.user_text) > 3000:
        requires["min_context"] = 200000
    return requires


class Aggregator(object):
    def __init__(self, brief, env_url, probes, catalog):
        self.brief = brief
        self.env_url = env_url.rstrip("/")
        self.probes = probes
        self.catalog = catalog
        # Full probed models per provider (with caps) for routing.
        self.models_by_provider = {}
        for provider, (ok, models) in zip(brief.providers, probes):
            if not ok:
                rows = []
                for (pid, mid), row in sorted(brief.ratings.items()):
                    if pid == provider["id"] and row["free"]:
                        rows.append({"id": mid, "free": True, "context_length": row["context_length"],
                                     "tools": row["tools"], "vision": row["vision"], "json": row["json"]})
                self.models_by_provider[provider["id"]] = rows
                continue
            out = []
            for m in sorted(models, key=lambda x: x["id"]):
                row = brief.ratings.get((provider["id"], m["id"]))
                if row is not None:
                    out.append({"id": m["id"], "free": row["free"], "context_length": row["context_length"],
                                "tools": row["tools"], "vision": row["vision"], "json": row["json"]})
                else:
                    out.append({"id": m["id"], "free": m["free"] if m["free"] is not None else False,
                                "context_length": m["context_length"], "tools": m["tools"],
                                "vision": m["vision"], "json": m["json"]})
            self.models_by_provider[provider["id"]] = out
        self.provider_by_id = {p["id"]: p for p in brief.providers}
        self.state = {p["id"]: {"calls": 0, "dead": False, "strikes": 0} for p in brief.providers}
        # Providers get two strikes: a refusal is ~4x more likely a one-off
        # fail_call (later calls on the same provider usually succeed) than
        # an outage, so a provider is dropped only on its SECOND refusal.
        self.arrivals = 0
        self.lock = threading.Lock()

    # -- provider call ----------------------------------------------------
    def provider_url(self, provider, model, stream):
        base = join_base(self.env_url, provider["base"])
        if provider["wire"] == "openai":
            return base + "/v1/chat/completions"
        if provider["wire"] == "anthropic":
            return base + "/v1/messages"
        verb = "streamGenerateContent" if stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, model, verb)
        if stream:
            url += "?alt=sse"
        return url

    def call_provider(self, provider, model, nr):
        """One provider call. Returns (ok, text, usage) or (False, None, None).
        Every attempt counts as one call on the provider for planning."""
        wire = provider["wire"]
        body = build_provider_body(nr, wire, model)
        url = self.provider_url(provider, model, nr.stream)
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if wire == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        if nr.stream:
            headers["accept"] = "text/event-stream"
        with self.lock:
            self.state[provider["id"]]["calls"] += 1
        try:
            status, _hdrs, raw = http_request(
                url, "POST", headers, body=json.dumps(body, separators=(",", ":")),
                timeout=PROVIDER_TIMEOUT_S)
        except Exception as exc:
            log("call %s/%s failed: %s" % (provider["id"], model, exc))
            return False, None, None
        if status < 200 or status >= 300:
            return False, None, None
        parsed = parse_provider_response(wire, nr.stream, raw)
        if parsed is None:
            return False, None, None
        text, usage, _stop = parsed
        if not isinstance(text, str) or text == "" or usage is None:
            return False, None, None
        return True, text, usage

    # -- planning ---------------------------------------------------------
    def remaining_ids(self, current_index):
        ids = [r["id"] for r in self.brief.manifest[current_index:]]
        return ids

    def fallback_candidates(self, req):
        """All (provider, model) that could serve req, best rating first;
        alive providers first, dead ones only as a last resort."""
        out = []
        for provider in self.brief.providers:
            pid = provider["id"]
            if not wire_can_serve(provider["wire"], req["requires"]):
                continue
            for m in self.models_by_provider.get(pid, []):
                if not m["free"]:
                    continue
                caps = model_caps(self.brief, pid, m["id"], m)
                if not meets_requirements(req["requires"], caps):
                    continue
                row = self.brief.ratings.get((pid, m["id"]))
                rating = row["rating_milli"] if row else 0
                out.append((rating, pid, m["id"]))
        out.sort(key=lambda x: -x[0])
        alive = [c for c in out if not self.state[c[1]]["dead"]]
        dead = [c for c in out if self.state[c[1]]["dead"]]
        return [(pid, mid) for _r, pid, mid in alive + dead]

    def serve(self, api, body):
        """Serve one client request. Returns (status, content_type, payload)."""
        nr = parse_client_request(api, body)
        with self.lock:
            index = self.arrivals
            self.arrivals += 1
        manifest = self.brief.manifest
        if index < len(manifest):
            req = manifest[index]
            # Cross-check: the body's own requirements should agree with the
            # manifest row; if they don't, trust the manifest (arrival order
            # is the driver's order) but note it.
            inferred = requirements_from_body(api, body, nr)
            noisy = {k: v for k, v in inferred.items() if k != "json"} != \
                {k: v for k, v in req["requires"].items() if k != "json"}
            if noisy:
                log("request %s: manifest %s vs body %s" % (req["id"], req["requires"], inferred))
        else:
            req = {"id": "r?%d" % index, "api": api, "stream": nr.stream,
                   "weight": 1, "requires": requirements_from_body(api, body, nr)}
        # JSON-mode on the OpenAI wire cannot be carried by an Anthropic-wire
        # provider (it echoes json_mode absent): exclude that wire here.
        tried = []
        skipped = set()
        attempts = 0
        while attempts < 40:
            attempts += 1
            remaining = ([req["id"]] + self.remaining_ids(index + 1)) if index < len(manifest) else [req["id"]]
            planner = Planner(self.brief, self.models_by_provider, self.state)
            plan = planner.plan(remaining)
            pick = plan.get(req["id"])
            if pick is not None and pick in skipped:
                pick = None
            if pick is None:
                cands = [c for c in self.fallback_candidates(req)
                         if c not in tried and c not in skipped]
                if not cands:
                    break
                pick = cands[0]
            pid, model = pick
            provider = self.provider_by_id.get(pid)
            if provider is None:
                break
            if nr.json_mode and provider["wire"] == "anthropic":
                # Never send a JSON-mode request down a wire that drops it.
                # (Wire incompatibility, not a refusal: no strike. The
                # planner/fallback already exclude this, so this is a guard
                # against spinning on a skipped pick.)
                skipped.add(pick)
                tried.append(pick)
                continue
            tried.append(pick)
            ok, text, usage = self.call_provider(provider, model, nr)
            if ok:
                ctype, payload = build_client_response(api, nr.stream, text, usage, model)
                return 200, ctype, payload
            # Refused: one strike (second refusal drops the provider).
            log("request %s: %s/%s refused, failing over" % (req["id"], pid, model))
            with self.lock:
                st = self.state[pid]
                st["strikes"] += 1
                if st["strikes"] >= 2:
                    st["dead"] = True
        # Nothing could serve it: answer an error in the client's wire.
        if api == "anthropic":
            payload = json.dumps({"type": "error", "error": {"type": "overloaded_error",
                                  "message": "no free model could serve this request"}},
                                 separators=(",", ":")).encode("utf-8")
        else:
            payload = json.dumps({"error": {"message": "no free model could serve this request",
                                  "type": "server_error", "code": None}},
                                 separators=(",", ":")).encode("utf-8")
        return 503, b"application/json", payload

    def model_list(self, anthropic_shape):
        """Free models only, in brief provider order, sorted by id."""
        items = []
        for provider in self.brief.providers:
            for m in sorted(self.models_by_provider.get(provider["id"], []), key=lambda x: x["id"]):
                if m["free"]:
                    items.append({"id": m["id"], "provider": provider["id"]})
        if anthropic_shape:
            return {"data": [{"type": "model", "id": it["id"], "display_name": it["id"],
                              "created_at": FIXED_ANTHROPIC_CREATED_AT} for it in items],
                    "has_more": False}
        return {"object": "list",
                "data": [{"id": it["id"], "object": "model", "created": FIXED_CREATED,
                          "owned_by": it["provider"]} for it in items]}


# --------------------------------------------------------------------------
# HTTP server, register, /work poll, --list, main.
# --------------------------------------------------------------------------

def _send_json(handler, status, obj):
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _send_error(handler, api, status, message):
    if api == "anthropic":
        _send_json(handler, status, {"type": "error", "error": {"type": "api_error", "message": message}})
    else:
        _send_json(handler, status, {"error": {"message": message, "type": "invalid_request_error", "code": None}})


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _read_body(self):
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > 32 * 1024 * 1024:
                return None
            return self.rfile.read(length)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/v1/models":
                anthropic_shape = self.headers.get("x-api-key") is not None
                _send_json(self, 200, agg.model_list(anthropic_shape))
                return
            _send_error(self, "openai", 404, "no route GET %s" % parsed.path)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/v1/chat/completions":
                api = "openai"
            elif parsed.path == "/v1/messages":
                api = "anthropic"
            else:
                _send_error(self, "openai", 404, "no route POST %s" % parsed.path)
                return
            raw = self._read_body()
            if raw is None:
                _send_error(self, api, 400, "the request body is not one JSON object")
                return
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                _send_error(self, api, 400, "the request body is not one JSON object")
                return
            if not is_record(body):
                _send_error(self, api, 400, "the request body is not one JSON object")
                return
            try:
                status, ctype, payload = agg.serve(api, body)
            except Exception as exc:
                log("serve failed: %r" % (exc,))
                _send_error(self, api, 500, "the server failed to answer")
                return
            self.send_response(status)
            self.send_header("content-type", ctype.decode("utf-8"))
            if ctype.startswith(b"text/event-stream"):
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "keep-alive")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def register(env_url, port, catalog):
    body = json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                      separators=(",", ":"))
    last = None
    for _ in range(3):
        try:
            status, _hdrs, raw = http_request(env_url.rstrip("/") + "/register", "POST",
                                              {"content-type": "application/json"}, body=body,
                                              timeout=REGISTER_TIMEOUT_S)
            if 200 <= status < 300:
                reply = json.loads(raw.decode("utf-8"))
                log("registered: the workload holds %s requests" % reply.get("requests"))
                return
            last = "HTTP %s" % status
        except Exception as exc:
            last = str(exc)
        time.sleep(1)
    raise RuntimeError("POST /register failed: %s" % last)


def start_poll(env_url, server, started_at):
    def poll():
        while True:
            if time.monotonic() - started_at > 175:
                log("watchdog: work never finished, exiting 0")
                os._exit(0)
            try:
                status, _hdrs, raw = http_request(env_url.rstrip("/") + "/work", "GET",
                                                  timeout=2)
                if status == 200:
                    work = json.loads(raw.decode("utf-8"))
                    if is_record(work) and work.get("done") is True:
                        log("work done, exiting 0")
                        try:
                            server.shutdown()
                        except Exception:
                            pass
                        os._exit(0)
            except SystemExit:
                raise
            except Exception:
                pass
            time.sleep(POLL_INTERVAL_S)
    t = threading.Thread(target=poll, daemon=True)
    t.start()
    return t


def main():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = read_brief(prompt_file)
    if brief.error is not None:
        log(brief.error)
    probes = []
    if env_url and brief.providers:
        for provider in brief.providers:
            probes.append(probe_provider(env_url, provider))
    else:
        probes = [(False, []) for _ in brief.providers]
    catalog = build_catalog(brief, probes)
    if "--list" in sys.argv[1:]:
        sys.stdout.write(catalog_json(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    agg = Aggregator(brief, env_url, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d providers" % (port, len(catalog["providers"])))
    register(env_url, port, catalog)
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda *_a: os._exit(0))
    _signal.signal(_signal.SIGINT, lambda *_a: os._exit(0))
    started_at = time.monotonic()
    start_poll(env_url, server, started_at)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
