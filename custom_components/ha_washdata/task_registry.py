# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""In-memory registry of long-running background tasks (reprocess, ML training,
Playground history/optimize).

Purpose: keep a task's progress, cancel handle and result on the *server* so they
survive a dropped WebSocket (backgrounded tab), can be cancelled, and can be
re-fetched on reconnect. One registry per ``hass``; each task is tagged with the
``entry_id`` it belongs to. No persistence - results live for the session and the
last few finished tasks are retained for reload.

Pure asyncio + synchronous listener callbacks; the WebSocket layer registers a
listener to push updates and calls :func:`get_registry` to read/kick/cancel.
"""
from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from homeassistant.core import HomeAssistant

from .const import DOMAIN

# Keep this many *finished* tasks (with their results) around for reload; older
# ones are evicted. Running tasks are never evicted.
_MAX_FINISHED = 30
_REGISTRY_KEY = f"{DOMAIN}_task_registry"

# Task lifecycle states.
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"


@dataclass
class Task:
    """A single tracked background operation."""

    id: str
    entry_id: str
    kind: str          # 'reprocess' | 'ml_training' | 'pg_history' | 'pg_sweep'
    label: str
    total: int = 0
    done: int = 0
    state: str = STATE_RUNNING
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: Any = None
    _cancelled: bool = False

    @property
    def cancel_requested(self) -> bool:
        return self._cancelled

    def progress(self) -> float | None:
        """Fraction complete in [0, 1], or None when total is unknown."""
        if self.total <= 0:
            return None
        return max(0.0, min(1.0, self.done / self.total))

    def eta_s(self) -> float | None:
        """Rough seconds-to-completion from elapsed time and progress."""
        p = self.progress()
        if not p or p <= 0 or self.state != STATE_RUNNING:
            return None
        elapsed = self.updated_at - self.started_at
        if elapsed <= 0:
            return None
        return max(0.0, elapsed * (1.0 - p) / p)

    def snapshot(self, include_result: bool = False) -> dict[str, Any]:
        """JSON-safe view for the WS layer. ``include_result`` embeds the payload."""
        data: dict[str, Any] = {
            "id": self.id,
            "entry_id": self.entry_id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "progress": self.progress(),
            "eta_s": self.eta_s(),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "has_result": self.result is not None,
        }
        if include_result:
            data["result"] = self.result
        return data


class TaskRegistry:
    """Holds active + recently-finished tasks and notifies listeners on change."""

    def __init__(self) -> None:
        self._tasks: OrderedDict[str, Task] = OrderedDict()
        self._listeners: set[Callable[[dict[str, Any]], None]] = set()

    # -- listeners -----------------------------------------------------------
    def add_listener(self, cb: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        """Register a change callback; returns an unsubscribe function."""
        self._listeners.add(cb)
        return lambda: self._listeners.discard(cb)

    def _notify(self, task: Task) -> None:
        snap = task.snapshot()
        for cb in list(self._listeners):
            try:
                cb(snap)
            except Exception:  # pylint: disable=broad-exception-caught
                # A misbehaving listener must never break task bookkeeping.
                pass

    # -- lifecycle -----------------------------------------------------------
    def create(self, entry_id: str, kind: str, label: str, total: int = 0) -> Task:
        task = Task(
            id=uuid.uuid4().hex[:12],
            entry_id=entry_id,
            kind=kind,
            label=label,
            total=max(0, int(total or 0)),
        )
        self._tasks[task.id] = task
        self._notify(task)
        self._evict()
        return task

    def update(
        self,
        task: Task,
        *,
        done: int | None = None,
        total: int | None = None,
        label: str | None = None,
    ) -> None:
        if done is not None:
            task.done = done
        if total is not None:
            task.total = total
        if label is not None:
            task.label = label
        task.updated_at = time.time()
        self._notify(task)

    def finish(
        self,
        task: Task,
        *,
        state: str = STATE_DONE,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        task.state = state
        task.error = error
        if result is not None:
            task.result = result
        task.finished_at = task.updated_at = time.time()
        self._notify(task)
        self._evict()

    def cancel(self, task_id: str) -> bool:
        """Request cancellation of a running task. Consumers poll
        :attr:`Task.cancel_requested` between chunks. Returns True if a running
        task was flagged."""
        task = self._tasks.get(task_id)
        if task is not None and task.state == STATE_RUNNING:
            task._cancelled = True  # noqa: SLF001 - registry owns the flag
            return True
        return False

    # -- reads ---------------------------------------------------------------
    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def snapshot(self, entry_id: str | None = None) -> list[dict[str, Any]]:
        return [
            t.snapshot()
            for t in self._tasks.values()
            if entry_id is None or t.entry_id == entry_id
        ]

    def _evict(self) -> None:
        finished = [tid for tid, t in self._tasks.items() if t.state != STATE_RUNNING]
        while len(finished) > _MAX_FINISHED:
            self._tasks.pop(finished.pop(0), None)


def get_registry(hass: HomeAssistant) -> TaskRegistry:
    """Get (or lazily create) the per-hass task registry."""
    reg = hass.data.get(_REGISTRY_KEY)
    if not isinstance(reg, TaskRegistry):
        reg = TaskRegistry()
        hass.data[_REGISTRY_KEY] = reg
    return reg
