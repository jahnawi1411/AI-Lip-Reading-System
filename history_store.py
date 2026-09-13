"""Shared persistence: SQLite locally, PostgreSQL for multiple replicas."""
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import (create_engine, MetaData, Table, Column, String, Integer,
                        Float, Text, select, delete, update)
from sqlalchemy.pool import StaticPool
import json
import time


class Store:
    def __init__(self, url):
        kwargs = {"pool_pre_ping": True}
        if url == "sqlite://":
            kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
        self.engine = create_engine(url, **kwargs)
        meta = MetaData()
        self.users = Table("users", meta, Column("id", String(32), primary_key=True),
                           Column("username", String(40), unique=True, nullable=False),
                           Column("password", Text, nullable=False))
        self.history = Table("history", meta, Column("id", String(32), primary_key=True),
                             Column("owner", String(32), nullable=False, index=True),
                             Column("created", String(40), nullable=False), Column("result", Text, nullable=False))
        self.limits = Table("request_limits", meta, Column("key", String(100), primary_key=True),
                            Column("window", Float, nullable=False), Column("count", Integer, nullable=False))
        meta.create_all(self.engine)

    def user(self, username=None, uid=None):
        with self.engine.connect() as con:
            clause = self.users.c.username == username if username is not None else self.users.c.id == uid
            row = con.execute(select(self.users).where(clause)).mappings().first()
            return dict(row) if row else None

    def register(self, username, password_hash):
        uid = uuid4().hex
        with self.engine.begin() as con:
            con.execute(self.users.insert().values(id=uid, username=username, password=password_hash))
        return self.user(uid=uid)

    def allow(self, key, maximum=10, seconds=60):
        # Database upsert makes the per-window increment atomic across replicas.
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        now = time.time(); window = now - now % seconds
        insert = pg_insert if self.engine.dialect.name == "postgresql" else sqlite_insert
        from sqlalchemy import case
        statement = insert(self.limits).values(key=key, window=window, count=1)
        statement = statement.on_conflict_do_update(index_elements=["key"], set_={
            "window": window, "count": case((self.limits.c.window == window, self.limits.c.count + 1), else_=1)
        }).returning(self.limits.c.count)
        with self.engine.begin() as con:
            count = con.execute(statement).scalar_one()
            con.execute(delete(self.limits).where(self.limits.c.window < now - 86400))
        return count <= maximum

    def save(self, owner, result):
        identifier = uuid4().hex
        created = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as con:
            con.execute(self.history.insert().values(id=identifier, owner=owner, created=created, result=json.dumps(result)))
            old = con.execute(select(self.history.c.id).where(self.history.c.owner == owner)
                              .order_by(self.history.c.created.desc()).offset(200)).scalars().all()
            if old:
                con.execute(delete(self.history).where(self.history.c.id.in_(old), self.history.c.owner == owner))
        return identifier

    def list(self, owner):
        with self.engine.connect() as con:
            rows = con.execute(select(self.history).where(self.history.c.owner == owner)
                               .order_by(self.history.c.created.desc()).limit(200)).mappings()
            return [{"id": r["id"], "created": r["created"], "result": json.loads(r["result"])} for r in rows]

    def remove(self, owner, identifier=None):
        clause = self.history.c.owner == owner
        if identifier is not None:
            clause &= self.history.c.id == identifier
        with self.engine.begin() as con:
            return con.execute(delete(self.history).where(clause)).rowcount
