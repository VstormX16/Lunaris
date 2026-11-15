"""Saved operations (History list and "pick up where you left off" cards) in SQLite.

One short-lived connection per call keeps it safe under FastAPI's thread pool.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

_SCHEMA = """CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    params TEXT NOT NULL,
    result TEXT NOT NULL
)"""


def _plain(o):
    """JSON fallback for NumPy scalars (np.int64, np.bool_, ...)."""
    return o.item() if hasattr(o, "item") else str(o)


class OperationStore:
    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            with c:  # commit on success, rollback on error
                yield c
        finally:
            c.close()  # sqlite3's own context manager does not close the connection

    def create(self, name, kind, params, result):
        op = {"id": uuid.uuid4().hex, "name": name, "kind": kind,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "params": params, "result": result}
        with self._conn() as c:
            c.execute("INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?)",
                      (op["id"], name, kind, op["created_at"], json.dumps(params, default=_plain),
                       json.dumps(result, default=_plain)))
        return op

    def list(self, kind=None, query=None, limit=50):
        """Newest first, without the (potentially large) result body."""
        sql, args = "SELECT id, name, kind, created_at FROM operations WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if query:
            sql += " AND name LIKE ?"
            args.append(f"%{query}%")
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args)]

    def get(self, op_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM operations WHERE id = ?", (op_id,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["params"], d["result"] = json.loads(d["params"]), json.loads(d["result"])
        return d

    def delete(self, op_id):
        with self._conn() as c:
            return c.execute("DELETE FROM operations WHERE id = ?", (op_id,)).rowcount > 0


_SESSIONS = """CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state TEXT NOT NULL,
    chat TEXT NOT NULL
)"""


class SessionStore(OperationStore):
    """What the History list shows: one saved New Operation (its inputs) with the assistant conversation that belongs to it."""

    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute(_SESSIONS)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def create_session(self, name, state, chat):
        sid, now = uuid.uuid4().hex, self._now()
        with self._conn() as c:
            c.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)", (sid, name, now, now, json.dumps(state), json.dumps(chat)))
        return self.get_session(sid)

    def list_sessions(self, limit=100):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT id, name, created_at, updated_at, json_extract(state, '$.kind') AS kind, json_extract(state, '$.a.vehicleId') AS vehicle, json_extract(state, '$.mode') AS mode FROM sessions ORDER BY updated_at DESC, rowid DESC LIMIT ?", (limit,))]

    def get_session(self, sid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
        if r is None:
            return None
        d = dict(r)
        d["state"], d["chat"] = json.loads(d["state"]), json.loads(d["chat"])
        return d

    def update_session(self, sid, name=None, state=None, chat=None):
        sets, args = ["updated_at = ?"], [self._now()]
        for col, val in (("name", name), ("state", None if state is None else json.dumps(state)), ("chat", None if chat is None else json.dumps(chat))):
            if val is not None:
                sets.append(f"{col} = ?")
                args.append(val)
        with self._conn() as c:
            n = c.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*args, sid)).rowcount
        return self.get_session(sid) if n else None

    def delete_session(self, sid):
        with self._conn() as c:
            return c.execute("DELETE FROM sessions WHERE id = ?", (sid,)).rowcount > 0


class CustomVehicleStore:
    """Vehicles designed in the garage (one JSON profile each), kept in the same SQLite file."""

    def __init__(self, path):
        self.path = str(path)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS custom_vehicles (id TEXT PRIMARY KEY, profile TEXT NOT NULL)")

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        try:
            with c:
                yield c
        finally:
            c.close()

    def all(self):
        with self._conn() as c:
            return [json.loads(r[0]) for r in c.execute("SELECT profile FROM custom_vehicles ORDER BY rowid")]

    def save(self, profile):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO custom_vehicles (id, profile) VALUES (?, ?)", (profile["id"], json.dumps(profile)))

    def delete(self, vid):
        with self._conn() as c:
            return c.execute("DELETE FROM custom_vehicles WHERE id = ?", (vid,)).rowcount > 0
