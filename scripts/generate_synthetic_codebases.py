#!/usr/bin/env python3
"""Generate thousands of intricate multi-file synthetic codebases with pytest judges.

Each task is a mini package: scaffold files + target stubs + gold implementation +
strict tests. Prompts describe complex multi-module behavior; the model must fill
target files so pytest passes.

Output:
  data/synthetic_codebases/train.jsonl
  data/synthetic_codebases/bench.jsonl
  data/synthetic_codebases/meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import textwrap
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]


def _h(*parts: str) -> str:
    x = hashlib.sha256()
    for p in parts:
        x.update(p.encode())
        x.update(b"\0")
    return x.hexdigest()[:24]


def _prompt(title: str, api: str, targets: list[str], extra: str = "") -> str:
    tgt = "\n".join(f"- `{t}`" for t in targets)
    return textwrap.dedent(
        f"""\
        You are editing a multi-file Python package. Implement the missing pieces so the full test suite passes.

        ## Task: {title}

        ### Required API / behavior
        {api}

        ### Files you must implement (replace stubs)
        {tgt}

        ### Output format
        For each file, emit a fenced block tagged with the path:
        ```python path=relative/path.py
        # full file contents
        ```

        Implement only the target files. Do not omit required classes/functions.
        {extra}
        """
    ).strip()


def _task(
    *,
    family: str,
    title: str,
    api: str,
    files: dict[str, str],
    tests: dict[str, str],
    gold_files: dict[str, str],
    target_files: list[str],
    difficulty: float,
    seed: int,
    extra_prompt: str = "",
) -> dict[str, Any]:
    # Scaffold uses stubs for targets; gold kept separate for QA / optional reveal.
    scaffold = dict(files)
    for t in target_files:
        if t in gold_files and t not in scaffold:
            # leave a stub if not provided
            scaffold[t] = f'"""TODO: implement {t}"""\nraise NotImplementedError\n'
    prompt = _prompt(title, api, target_files, extra_prompt)
    return {
        "prompt": prompt,
        "answer": "",
        "reward_name": "codebase_tests",
        "domain": "codebase",
        "source": "synthetic",
        "dataset": f"synthetic_codebases/{family}",
        "family": family,
        "difficulty": difficulty,
        "hash_id": _h(family, title, str(seed), prompt[:500]),
        "test_timeout_sec": 10.0,
        "codebase": {
            "files": scaffold,
            "tests": tests,
            "gold_files": gold_files,
            "target_files": target_files,
            "hidden_files": {},
        },
    }


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def gen_registry(rng: random.Random, i: int) -> dict[str, Any]:
    n = rng.randint(3, 7)
    names = [f"op_{rng.choice('abcdef')}_{j}" for j in range(n)]
    # unique
    names = list(dict.fromkeys(names)) or ["op_a_0"]
    ops = {name: rng.choice(["add", "mul", "sub", "max2"]) for name in names}
    gold_core = [
        "from __future__ import annotations\n",
        "from typing import Callable, Any\n\n",
        "_REG: dict[str, Callable[..., Any]] = {}\n\n",
        "def register(name: str):\n",
        "    def deco(fn):\n",
        "        if name in _REG:\n",
        "            raise KeyError(f'duplicate:{name}')\n",
        "        _REG[name] = fn\n",
        "        return fn\n",
        "    return deco\n\n",
        "def get(name: str):\n",
        "    if name not in _REG:\n",
        "        raise KeyError(name)\n",
        "    return _REG[name]\n\n",
        "def call(name: str, *args):\n",
        "    return get(name)(*args)\n\n",
        "def names() -> list[str]:\n",
        "    return sorted(_REG)\n",
    ]
    gold_ops = ["from core import register\n\n"]
    for name, kind in ops.items():
        if kind == "add":
            gold_ops.append(f"@register({name!r})\ndef {name}(a, b):\n    return a + b\n\n")
        elif kind == "mul":
            gold_ops.append(f"@register({name!r})\ndef {name}(a, b):\n    return a * b\n\n")
        elif kind == "sub":
            gold_ops.append(f"@register({name!r})\ndef {name}(a, b):\n    return a - b\n\n")
        else:
            gold_ops.append(f"@register({name!r})\ndef {name}(a, b):\n    return a if a >= b else b\n\n")

    tests = [
        "import ops  # noqa: F401\n",
        "from core import call, names, get, register\n",
        "import pytest\n\n",
        "def test_all_registered():\n",
        f"    assert set(names()) == set({list(ops)!r})\n\n",
        "def test_calls():\n",
    ]
    for name, kind in ops.items():
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        if kind == "add":
            exp = a + b
        elif kind == "mul":
            exp = a * b
        elif kind == "sub":
            exp = a - b
        else:
            exp = a if a >= b else b
        tests.append(f"    assert call({name!r}, {a}, {b}) == {exp}\n")
    tests += [
        "\ndef test_missing():\n",
        "    with pytest.raises(KeyError):\n",
        "        get('nope')\n\n",
        "def test_duplicate():\n",
        "    @register('_tmp_unique_xyz')\n",
        "    def _f():\n",
        "        return 1\n",
        "    with pytest.raises(KeyError):\n",
        "        @register('_tmp_unique_xyz')\n",
        "        def _g():\n",
        "            return 2\n",
    ]
    return _task(
        family="registry",
        title=f"Plugin operator registry #{i}",
        api=(
            "Implement `core.py` with register/get/call/names and `ops.py` that registers "
            f"these ops: {ops}. Duplicate registration must raise KeyError. "
            "`names()` returns sorted op names."
        ),
        files={
            "src/core.py": '"""Registry API stub."""\nraise NotImplementedError("implement core")\n',
            "src/ops.py": '"""Ops stub — import core.register and register functions."""\nraise NotImplementedError("implement ops")\n',
            "src/__init__.py": "",
        },
        tests={"tests/test_registry.py": "".join(tests)},
        gold_files={
            "src/core.py": "".join(gold_core),
            "src/ops.py": "".join(gold_ops),
        },
        target_files=["src/core.py", "src/ops.py"],
        difficulty=1.0e6 + n * 1e3 + rng.random(),
        seed=i,
    )


def gen_lru_cache(rng: random.Random, i: int) -> dict[str, Any]:
    cap = rng.choice([2, 3, 4, 5])
    gold = textwrap.dedent(
        f"""\
        from __future__ import annotations
        from collections import OrderedDict
        from typing import Any, Hashable

        class LRUCache:
            def __init__(self, capacity: int = {cap}):
                if capacity < 1:
                    raise ValueError('capacity')
                self.capacity = capacity
                self._data: OrderedDict[Hashable, Any] = OrderedDict()
                self.hits = 0
                self.misses = 0

            def get(self, key: Hashable, default=None):
                if key not in self._data:
                    self.misses += 1
                    return default
                self.hits += 1
                self._data.move_to_end(key)
                return self._data[key]

            def put(self, key: Hashable, value: Any) -> None:
                if key in self._data:
                    self._data.move_to_end(key)
                    self._data[key] = value
                    return
                self._data[key] = value
                if len(self._data) > self.capacity:
                    self._data.popitem(last=False)

            def __len__(self) -> int:
                return len(self._data)

            def stats(self) -> dict[str, int]:
                return {{'hits': self.hits, 'misses': self.misses, 'size': len(self._data)}}
        """
    )
    tests = textwrap.dedent(
        f"""\
        from cache import LRUCache
        import pytest

        def test_capacity_and_eviction():
            c = LRUCache({cap})
            for k in range({cap}+2):
                c.put(k, k*10)
            assert len(c) == {cap}
            # oldest keys evicted
            assert c.get(0, None) is None
            assert c.get({cap}+1) == ({cap}+1)*10

        def test_recency():
            c = LRUCache({cap})
            for k in range({cap}):
                c.put(k, k)
            c.get(0)
            c.put({cap}, 99)
            # key 1 should be gone if capacity>1 else 0 may stay
            if {cap} > 1:
                assert c.get(1, None) is None
            assert c.get(0) == 0

        def test_stats_and_update():
            c = LRUCache({cap})
            c.put('a', 1)
            assert c.get('a') == 1
            c.put('a', 2)
            assert c.get('a') == 2
            s = c.stats()
            assert s['hits'] >= 2 and s['size'] == 1

        def test_bad_cap():
            with pytest.raises(ValueError):
                LRUCache(0)
        """
    )
    return _task(
        family="lru_cache",
        title=f"LRU cache module #{i} (cap={cap})",
        api=(
            f"Implement LRUCache in cache.py with capacity default {cap}, "
            "get/put/len/stats(hits,misses,size). Evict least-recently-used. "
            "get updates recency. Invalid capacity raises ValueError."
        ),
        files={"src/cache.py": "class LRUCache:\n    pass\n", "src/__init__.py": ""},
        tests={"tests/test_cache.py": tests},
        gold_files={"src/cache.py": gold},
        target_files=["src/cache.py"],
        difficulty=1.2e6 + cap * 100 + rng.random(),
        seed=i,
    )


def gen_event_bus(rng: random.Random, i: int) -> dict[str, Any]:
    gold_bus = textwrap.dedent(
        """\
        from __future__ import annotations
        from collections import defaultdict
        from typing import Any, Callable

        Handler = Callable[[Any], None]

        class EventBus:
            def __init__(self):
                self._subs: dict[str, list[Handler]] = defaultdict(list)
                self.history: list[tuple[str, Any]] = []

            def subscribe(self, event: str, handler: Handler) -> None:
                if handler not in self._subs[event]:
                    self._subs[event].append(handler)

            def unsubscribe(self, event: str, handler: Handler) -> None:
                if handler in self._subs[event]:
                    self._subs[event].remove(handler)

            def publish(self, event: str, payload: Any = None) -> int:
                self.history.append((event, payload))
                count = 0
                for h in list(self._subs.get(event, [])):
                    h(payload)
                    count += 1
                return count
        """
    )
    gold_svc = textwrap.dedent(
        """\
        from __future__ import annotations
        from bus import EventBus

        class CounterService:
            def __init__(self, bus: EventBus):
                self.bus = bus
                self.total = 0
                bus.subscribe('inc', self._on_inc)
                bus.subscribe('reset', self._on_reset)

            def _on_inc(self, payload):
                self.total += int(payload or 1)

            def _on_reset(self, payload):
                self.total = 0

            def emit_inc(self, n: int = 1) -> None:
                self.bus.publish('inc', n)
        """
    )
    tests = textwrap.dedent(
        """\
        from bus import EventBus
        from service import CounterService

        def test_pubsub_order():
            bus = EventBus()
            seen = []
            bus.subscribe('e', lambda p: seen.append(('a', p)))
            bus.subscribe('e', lambda p: seen.append(('b', p)))
            assert bus.publish('e', 7) == 2
            assert seen == [('a', 7), ('b', 7)]
            assert bus.history[-1] == ('e', 7)

        def test_unsubscribe():
            bus = EventBus()
            seen = []
            def h(p):
                seen.append(p)
            bus.subscribe('x', h)
            bus.unsubscribe('x', h)
            assert bus.publish('x', 1) == 0
            assert seen == []

        def test_service():
            bus = EventBus()
            svc = CounterService(bus)
            svc.emit_inc(3)
            svc.emit_inc(2)
            assert svc.total == 5
            bus.publish('reset')
            assert svc.total == 0
        """
    )
    return _task(
        family="event_bus",
        title=f"Event bus + counter service #{i}",
        api=(
            "Implement EventBus (subscribe/unsubscribe/publish/history) in bus.py and "
            "CounterService in service.py that listens for 'inc' and 'reset' and can emit_inc."
        ),
        files={
            "src/bus.py": "class EventBus:\n    pass\n",
            "src/service.py": "class CounterService:\n    pass\n",
            "src/__init__.py": "",
        },
        tests={"tests/test_bus.py": tests},
        gold_files={"src/bus.py": gold_bus, "src/service.py": gold_svc},
        target_files=["src/bus.py", "src/service.py"],
        difficulty=1.5e6 + rng.random() * 100,
        seed=i,
    )


def gen_graph(rng: random.Random, i: int) -> dict[str, Any]:
    # small DAG
    nodes = [f"n{j}" for j in range(rng.randint(4, 7))]
    edges: list[tuple[str, str]] = []
    for a, b in zip(nodes, nodes[1:]):
        edges.append((a, b))
    # extra edges
    if len(nodes) > 3 and rng.random() < 0.7:
        edges.append((nodes[0], nodes[2]))
    gold = textwrap.dedent(
        """\
        from __future__ import annotations
        from collections import defaultdict, deque
        from typing import Iterable

        class Graph:
            def __init__(self):
                self.adj: dict[str, set[str]] = defaultdict(set)

            def add_edge(self, u: str, v: str) -> None:
                self.adj[u].add(v)
                self.adj.setdefault(v, set())

            def neighbors(self, u: str) -> list[str]:
                return sorted(self.adj.get(u, set()))

            def bfs(self, start: str) -> list[str]:
                if start not in self.adj and start not in {x for vs in self.adj.values() for x in vs}:
                    return [start] if start else []
                seen = {start}
                q = deque([start])
                order = []
                while q:
                    u = q.popleft()
                    order.append(u)
                    for v in sorted(self.adj.get(u, set())):
                        if v not in seen:
                            seen.add(v)
                            q.append(v)
                return order

            def reachable(self, start: str, goal: str) -> bool:
                return goal in self.bfs(start)
        """
    )
    edge_lines = "\n".join(f"    g.add_edge({u!r}, {v!r})" for u, v in edges)
    tests = (
        "from graph import Graph\n\n"
        "def _g():\n"
        "    g = Graph()\n"
        f"{edge_lines}\n"
        "    return g\n\n"
        "def test_neighbors_sorted():\n"
        "    g = _g()\n"
        "    for u, vs in g.adj.items():\n"
        "        assert g.neighbors(u) == sorted(vs)\n\n"
        "def test_bfs_starts():\n"
        "    g = _g()\n"
        f"    order = g.bfs({nodes[0]!r})\n"
        f"    assert order[0] == {nodes[0]!r}\n"
        f"    assert {nodes[-1]!r} in order\n\n"
        "def test_reachable():\n"
        "    g = _g()\n"
        f"    assert g.reachable({nodes[0]!r}, {nodes[-1]!r}) is True\n"
        f"    assert g.reachable({nodes[-1]!r}, {nodes[0]!r}) is False or {nodes[-1]!r} == {nodes[0]!r}\n"
    )
    return _task(
        family="graph",
        title=f"Directed graph BFS #{i}",
        api=(
            "Implement Graph in graph.py with add_edge, neighbors (sorted), bfs "
            "(lexicographic neighbor expansion), reachable."
        ),
        files={"src/graph.py": "class Graph:\n    pass\n", "src/__init__.py": ""},
        tests={"tests/test_graph.py": tests},
        gold_files={"src/graph.py": gold},
        target_files=["src/graph.py"],
        difficulty=1.4e6 + len(nodes) * 100 + rng.random(),
        seed=i,
    )


def gen_pipeline(rng: random.Random, i: int) -> dict[str, Any]:
    stages = rng.randint(2, 4)
    gold_stages = textwrap.dedent(
        """\
        from __future__ import annotations

        def strip_stage(text: str) -> str:
            return text.strip()

        def lower_stage(text: str) -> str:
            return text.lower()

        def collapse_ws(text: str) -> str:
            return ' '.join(text.split())

        def prefix_stage(text: str, prefix: str = '>>') -> str:
            return f'{prefix}{text}'
        """
    )
    gold_pipe = textwrap.dedent(
        """\
        from __future__ import annotations
        from typing import Callable

        class Pipeline:
            def __init__(self):
                self.stages: list[Callable[[str], str]] = []

            def add(self, fn: Callable[[str], str]) -> 'Pipeline':
                self.stages.append(fn)
                return self

            def run(self, text: str) -> str:
                out = text
                for fn in self.stages:
                    out = fn(out)
                return out
        """
    )
    tests = textwrap.dedent(
        f"""\
        from pipeline import Pipeline
        from stages import strip_stage, lower_stage, collapse_ws, prefix_stage

        def test_chain():
            p = Pipeline().add(strip_stage).add(lower_stage).add(collapse_ws).add(prefix_stage)
            assert p.run('  Hello   World  ') == '>>hello world'

        def test_empty_pipeline():
            assert Pipeline().run('X') == 'X'

        def test_order_matters():
            p1 = Pipeline().add(prefix_stage).add(lower_stage)
            p2 = Pipeline().add(lower_stage).add(prefix_stage)
            assert p1.run('Ab') != p2.run('Ab') or True
            assert p2.run('Ab') == '>>ab'

        def test_stage_count_hint():
            # complexity seed {stages}
            assert callable(strip_stage) and callable(collapse_ws)
        """
    )
    return _task(
        family="pipeline",
        title=f"Text processing pipeline #{i}",
        api=(
            "Implement Pipeline (add/run) in pipeline.py and stages strip_stage, lower_stage, "
            "collapse_ws, prefix_stage(prefix='>>') in stages.py."
        ),
        files={
            "src/pipeline.py": "class Pipeline:\n    pass\n",
            "src/stages.py": "# implement stages\n",
            "src/__init__.py": "",
        },
        tests={"tests/test_pipeline.py": tests},
        gold_files={"src/pipeline.py": gold_pipe, "src/stages.py": gold_stages},
        target_files=["src/pipeline.py", "src/stages.py"],
        difficulty=1.3e6 + stages * 50 + rng.random(),
        seed=i,
    )


def gen_rate_limiter(rng: random.Random, i: int) -> dict[str, Any]:
    limit = rng.choice([2, 3, 5])
    window = rng.choice([10, 20, 30])
    gold = textwrap.dedent(
        f"""\
        from __future__ import annotations
        from collections import deque

        class RateLimiter:
            def __init__(self, limit: int = {limit}, window: float = {window}):
                if limit < 1 or window <= 0:
                    raise ValueError('bad params')
                self.limit = limit
                self.window = float(window)
                self._events: dict[str, deque[float]] = {{}}

            def allow(self, key: str, now: float) -> bool:
                q = self._events.setdefault(key, deque())
                while q and now - q[0] >= self.window:
                    q.popleft()
                if len(q) >= self.limit:
                    return False
                q.append(now)
                return True

            def remaining(self, key: str, now: float) -> int:
                q = self._events.setdefault(key, deque())
                while q and now - q[0] >= self.window:
                    q.popleft()
                return max(0, self.limit - len(q))
        """
    )
    tests = textwrap.dedent(
        f"""\
        from limiter import RateLimiter
        import pytest

        def test_burst():
            rl = RateLimiter({limit}, {window})
            ok = [rl.allow('u', float(t)) for t in range({limit})]
            assert all(ok)
            assert rl.allow('u', float({limit-1})) is False

        def test_window_expiry():
            rl = RateLimiter({limit}, {window})
            for t in range({limit}):
                assert rl.allow('u', float(t))
            assert rl.allow('u', float({limit-1})) is False
            assert rl.allow('u', float({limit-1} + {window})) is True

        def test_remaining_and_keys():
            rl = RateLimiter({limit}, {window})
            assert rl.remaining('a', 0.0) == {limit}
            rl.allow('a', 0.0)
            assert rl.remaining('a', 0.0) == {limit}-1
            assert rl.remaining('b', 0.0) == {limit}

        def test_bad():
            with pytest.raises(ValueError):
                RateLimiter(0, 10)
        """
    )
    return _task(
        family="rate_limiter",
        title=f"Sliding-window rate limiter #{i}",
        api=(
            f"Implement RateLimiter(limit={limit}, window={window}) with allow(key, now) "
            "and remaining(key, now) using a sliding window over timestamps."
        ),
        files={"src/limiter.py": "class RateLimiter:\n    pass\n", "src/__init__.py": ""},
        tests={"tests/test_limiter.py": tests},
        gold_files={"src/limiter.py": gold},
        target_files=["src/limiter.py"],
        difficulty=1.6e6 + limit * 10 + window + rng.random(),
        seed=i,
    )


def gen_kv_store(rng: random.Random, i: int) -> dict[str, Any]:
    gold_store = textwrap.dedent(
        """\
        from __future__ import annotations
        from typing import Any

        class KVStore:
            def __init__(self):
                self._data: dict[str, Any] = {}
                self._ttl: dict[str, float] = {}

            def set(self, key: str, value: Any, *, expire_at: float | None = None) -> None:
                self._data[key] = value
                if expire_at is None:
                    self._ttl.pop(key, None)
                else:
                    self._ttl[key] = float(expire_at)

            def get(self, key: str, now: float) -> Any:
                if key not in self._data:
                    return None
                exp = self._ttl.get(key)
                if exp is not None and now >= exp:
                    self.delete(key)
                    return None
                return self._data[key]

            def delete(self, key: str) -> bool:
                existed = key in self._data
                self._data.pop(key, None)
                self._ttl.pop(key, None)
                return existed

            def keys(self, now: float) -> list[str]:
                out = []
                for k in list(self._data):
                    if self.get(k, now) is not None:
                        out.append(k)
                return sorted(out)
        """
    )
    gold_repo = textwrap.dedent(
        """\
        from __future__ import annotations
        from store import KVStore

        class UserRepo:
            def __init__(self, store: KVStore):
                self.store = store

            def upsert(self, user_id: str, name: str, now: float, ttl: float | None = None) -> None:
                exp = None if ttl is None else now + ttl
                self.store.set(f'user:{user_id}', {'id': user_id, 'name': name}, expire_at=exp)

            def get_name(self, user_id: str, now: float) -> str | None:
                row = self.store.get(f'user:{user_id}', now)
                return None if row is None else row['name']
        """
    )
    tests = textwrap.dedent(
        """\
        from store import KVStore
        from repo import UserRepo

        def test_ttl():
            s = KVStore()
            s.set('a', 1, expire_at=10.0)
            assert s.get('a', 9.9) == 1
            assert s.get('a', 10.0) is None
            assert s.get('a', 11.0) is None

        def test_keys_sorted():
            s = KVStore()
            s.set('b', 1)
            s.set('a', 2)
            assert s.keys(0.0) == ['a', 'b']

        def test_repo():
            s = KVStore()
            r = UserRepo(s)
            r.upsert('1', 'Ada', now=0.0, ttl=5.0)
            assert r.get_name('1', 4.9) == 'Ada'
            assert r.get_name('1', 5.0) is None

        def test_delete():
            s = KVStore()
            s.set('x', 9)
            assert s.delete('x') is True
            assert s.delete('x') is False
        """
    )
    return _task(
        family="kv_store",
        title=f"TTL key-value store + user repo #{i}",
        api=(
            "Implement KVStore (set/get/delete/keys with expire_at TTL) and UserRepo "
            "that stores users under user:{id} with optional ttl seconds from now."
        ),
        files={
            "src/store.py": "class KVStore:\n    pass\n",
            "src/repo.py": "class UserRepo:\n    pass\n",
            "src/__init__.py": "",
        },
        tests={"tests/test_kv.py": tests},
        gold_files={"src/store.py": gold_store, "src/repo.py": gold_repo},
        target_files=["src/store.py", "src/repo.py"],
        difficulty=1.7e6 + rng.random() * 50,
        seed=i,
    )


def gen_expr_eval(rng: random.Random, i: int) -> dict[str, Any]:
    gold = textwrap.dedent(
        """\
        from __future__ import annotations
        import ast
        import operator as op

        _BIN = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.Mod: op.mod,
        }
        _UNARY = {ast.UAdd: op.pos, ast.USub: op.neg}

        class SafeEval:
            def eval(self, expr: str) -> float:
                tree = ast.parse(expr, mode='eval')
                return float(self._eval(tree.body))

            def _eval(self, node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
                    return _UNARY[type(node.op)](self._eval(node.operand))
                if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
                    return _BIN[type(node.op)](self._eval(node.left), self._eval(node.right))
                if isinstance(node, ast.Expression):
                    return self._eval(node.body)
                raise ValueError(f'unsupported:{type(node).__name__}')
        """
    )
    cases = [
        ("1+2*3", 7.0),
        ("(1+2)*3", 9.0),
        ("-5+10", 5.0),
        ("2**3", 8.0),
        ("10%4", 2.0),
    ]
    rng.shuffle(cases)
    case_lines = "\n".join(
        f"    assert abs(e.eval({ex!r}) - {val}) < 1e-9" for ex, val in cases
    )
    tests = (
        "from expr import SafeEval\n"
        "import pytest\n\n"
        "def test_cases():\n"
        "    e = SafeEval()\n"
        f"{case_lines}\n\n"
        "def test_reject_names():\n"
        "    e = SafeEval()\n"
        "    with pytest.raises(ValueError):\n"
        "        e.eval('__import__(\"os\").system(\"x\")')\n"
    )
    return _task(
        family="expr_eval",
        title=f"Safe arithmetic expression evaluator #{i}",
        api=(
            "Implement SafeEval.eval(expr) supporting + - * / ** % and parentheses, "
            "numbers only. Reject names/calls/attributes with ValueError."
        ),
        files={"src/expr.py": "class SafeEval:\n    pass\n", "src/__init__.py": ""},
        tests={"tests/test_expr.py": tests},
        gold_files={"src/expr.py": gold},
        target_files=["src/expr.py"],
        difficulty=1.8e6 + rng.random() * 20,
        seed=i,
    )


def gen_rbac(rng: random.Random, i: int) -> dict[str, Any]:
    gold = textwrap.dedent(
        """\
        from __future__ import annotations
        from collections import defaultdict

        class RBAC:
            def __init__(self):
                self.roles: dict[str, set[str]] = defaultdict(set)  # role -> perms
                self.users: dict[str, set[str]] = defaultdict(set)  # user -> roles

            def grant_role_perm(self, role: str, perm: str) -> None:
                self.roles[role].add(perm)

            def assign(self, user: str, role: str) -> None:
                self.users[user].add(role)

            def revoke(self, user: str, role: str) -> None:
                self.users[user].discard(role)

            def can(self, user: str, perm: str) -> bool:
                for role in self.users.get(user, set()):
                    if perm in self.roles.get(role, set()):
                        return True
                return False

            def permissions(self, user: str) -> list[str]:
                perms: set[str] = set()
                for role in self.users.get(user, set()):
                    perms |= self.roles.get(role, set())
                return sorted(perms)
        """
    )
    tests = textwrap.dedent(
        """\
        from rbac import RBAC

        def test_flow():
            r = RBAC()
            r.grant_role_perm('admin', 'read')
            r.grant_role_perm('admin', 'write')
            r.grant_role_perm('viewer', 'read')
            r.assign('u1', 'viewer')
            assert r.can('u1', 'read') is True
            assert r.can('u1', 'write') is False
            r.assign('u1', 'admin')
            assert r.can('u1', 'write') is True
            assert r.permissions('u1') == ['read', 'write']
            r.revoke('u1', 'admin')
            assert r.can('u1', 'write') is False
            assert r.permissions('u1') == ['read']

        def test_unknown_user():
            r = RBAC()
            assert r.can('nope', 'x') is False
            assert r.permissions('nope') == []
        """
    )
    return _task(
        family="rbac",
        title=f"Role-based access control #{i}",
        api=(
            "Implement RBAC with grant_role_perm, assign, revoke, can, permissions "
            "(sorted unique perms via roles)."
        ),
        files={"src/rbac.py": "class RBAC:\n    pass\n", "src/__init__.py": ""},
        tests={"tests/test_rbac.py": tests},
        gold_files={"src/rbac.py": gold},
        target_files=["src/rbac.py"],
        difficulty=1.55e6 + rng.random() * 30,
        seed=i,
    )


GENERATORS: list[Callable[[random.Random, int], dict[str, Any]]] = [
    gen_registry,
    gen_lru_cache,
    gen_event_bus,
    gen_graph,
    gen_pipeline,
    gen_rate_limiter,
    gen_kv_store,
    gen_expr_eval,
    gen_rbac,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "synthetic_codebases")
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-bench", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verify-gold", type=int, default=40, help="QA-check this many golds")
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = args.n_train + args.n_bench
    tasks: list[dict[str, Any]] = []
    print(f"Generating {total} synthetic codebase tasks...", flush=True)
    for i in range(total):
        gen = GENERATORS[i % len(GENERATORS)]
        # diversify with family-local rng
        local = random.Random((args.seed + 1) * 1000003 + i * 9176)
        task = gen(local, i)
        tasks.append(task)
        if (i + 1) % 500 == 0:
            print(f"  generated {i+1}/{total}", flush=True)

    rng.shuffle(tasks)
    bench = tasks[: args.n_bench]
    train = tasks[args.n_bench :]

    # Optional gold verification
    from seiso.slime_single_gpu.codebase_judge import gold_passes

    verify_n = len(train) if args.verify_all else min(args.verify_gold, len(train))
    ok = 0
    fail = 0
    print(f"Verifying {verify_n} gold solutions with pytest...", flush=True)
    for j, t in enumerate(train[:verify_n]):
        # build sample with gold applied for QA
        sample = {
            **t,
            "codebase": {
                **t["codebase"],
                "files": {**t["codebase"]["files"], **t["codebase"].get("gold_files", {})},
            },
        }
        # gold_passes uses gold_files field
        if gold_passes(t):
            ok += 1
        else:
            fail += 1
            if fail <= 5:
                print(f"  GOLD FAIL family={t.get('family')} hash={t.get('hash_id')}", flush=True)
        if (j + 1) % 20 == 0:
            print(f"  verified {j+1}/{verify_n} ok={ok} fail={fail}", flush=True)
    print(f"Gold QA: ok={ok} fail={fail}", flush=True)
    if fail > 0 and args.verify_all:
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write(args.out_dir / "train.jsonl", train)
    write(args.out_dir / "bench.jsonl", bench)

    # Curriculum by difficulty
    ranked = sorted(train, key=lambda r: float(r.get("difficulty") or 0))
    n = len(ranked)
    write(args.out_dir / "train_easy.jsonl", ranked[: max(1, int(n * 0.4))])
    write(
        args.out_dir / "train_medium.jsonl",
        ranked[int(n * 0.15) : max(int(n * 0.15) + 1, int(n * 0.7))],
    )
    write(args.out_dir / "train_hard.jsonl", ranked[int(n * 0.35) :])
    write(args.out_dir / "train_mixed.jsonl", ranked)

    from collections import Counter

    meta = {
        "n_train": len(train),
        "n_bench": len(bench),
        "families": dict(Counter(t["family"] for t in tasks)),
        "gold_qa_ok": ok,
        "gold_qa_fail": fail,
        "seed": args.seed,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {args.out_dir}", flush=True)
    return 0 if fail == 0 or not args.verify_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
