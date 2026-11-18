"""Issue #532: Home Assistant 2026.9 gave a device a single ``config_entry_id``.

``DeviceEntry.config_entries`` became a deprecated shim that logs "Detected that
custom integration 'ha_washdata' accesses DeviceEntry.config_entries" on every
read and stops working in 2027.10. The integration reads a device's entries
through ``const.device_entry_ids``, which uses ``config_entry_id`` where it
exists and ``config_entries`` only on releases without it (the test environment
runs 2026.2.3, which has only ``config_entries``).
"""
from __future__ import annotations

import re
from pathlib import Path

from custom_components.ha_washdata.const import device_entry_ids

_SRC = Path(__file__).resolve().parents[1] / "custom_components" / "ha_washdata"


class _NewDevice:
    """A 2026.9+ device entry: reading the shim is the warning."""

    config_entry_id = "entry-1"

    @property
    def config_entries(self) -> set[str]:
        raise AssertionError("read the deprecated DeviceEntry.config_entries")


class _OldDevice:
    config_entries = {"entry-1", "entry-2"}


def test_a_new_device_is_read_by_its_config_entry_id() -> None:
    assert device_entry_ids(_NewDevice()) == ["entry-1"]


def test_an_old_device_is_read_by_its_config_entries() -> None:
    assert sorted(device_entry_ids(_OldDevice())) == ["entry-1", "entry-2"]


def test_nothing_usable_is_an_empty_list() -> None:
    assert device_entry_ids(None) == []
    assert device_entry_ids(object()) == []


def test_no_module_reads_the_deprecated_attribute() -> None:
    """Every device lookup goes through the helper (docstrings and comments aside)."""
    hits = []
    for path in sorted(_SRC.glob("*.py")):
        if path.name == "const.py":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "``" in code:
                continue
            if re.search(r"\b(?!hass\b|self\b)\w+\.config_entries\b(?!\.)", code) and "import" not in code:
                hits.append(f"{path.name}:{n}: {line.strip()}")
    assert hits == []


class _Dev:
    def __init__(self, dev_id: str, entry_id: str, composite: str | None = None) -> None:
        self.id, self.config_entry_id, self.composite_device_id = dev_id, entry_id, composite


class _Registry:
    def __init__(self, *devices: _Dev) -> None:
        self.devices = {d.id: d for d in devices}


def test_a_pre_split_device_id_still_finds_washdata_s_entry() -> None:
    """PR #529 review: HA 2026.9 split a device shared by a plug's entry and
    WashData's into one per entry; the old id resolves to a composite whose
    ``config_entry_id`` is only the former primary (the plug's)."""
    composite = _Dev("old-id", "plug-entry")
    registry = _Registry(
        _Dev("split-1", "plug-entry", composite="old-id"),
        _Dev("split-2", "wd-entry", composite="old-id"),
        _Dev("other", "x-entry"),
    )
    assert device_entry_ids(composite) == ["plug-entry"]
    assert device_entry_ids(composite, registry) == ["plug-entry", "wd-entry"]
    assert device_entry_ids(_Dev("plain", "wd-entry"), registry) == ["wd-entry"]


def test_every_device_lookup_passes_the_registry() -> None:
    calls = []
    for path in (_SRC / "__init__.py", _SRC / "device_trigger.py"):
        calls += re.findall(r"device_entry_ids\(([^)]*)\)", path.read_text(encoding="utf-8"))
    assert calls and all("," in args for args in calls), calls
