"""Audit 2026-10-02 PERF-02: entities are pushed, not polled.

No entity set ``should_poll = False``, so Home Assistant re-wrote all of them
every 30 s, idle included (188 ms per poll on a 13-profile store). Every entity
already subscribes to the manager's update signal; only values that advance
with the clock alone keep polling.
"""

from __future__ import annotations

import inspect

from homeassistant.helpers.entity import Entity

from custom_components.ha_washdata import binary_sensor, button, select, sensor

CLOCK_DRIVEN = {"WasherElapsedTimeSensor", "PumpRunsTodaySensor"}


def _entity_classes():
    for module in (sensor, binary_sensor, select, button):
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and issubclass(cls, Entity):
                yield name, cls


def _should_poll(cls: type) -> bool:
    # HA's CachedProperties metaclass moves a class-level `_attr_x` value to
    # `__attr_x` and leaves a property in its place; walk the MRO for the value.
    for klass in cls.__mro__:
        if "__attr_should_poll" in vars(klass):
            return bool(vars(klass)["__attr_should_poll"])
    return True


def test_only_clock_driven_entities_poll():
    polling = {name for name, cls in _entity_classes() if _should_poll(cls)}
    assert polling == CLOCK_DRIVEN
