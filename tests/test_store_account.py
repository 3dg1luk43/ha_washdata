"""Phase D: store account persistence + credential redaction."""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata import store_account
from custom_components.ha_washdata.diagnostics import _redact
from custom_components.ha_washdata.profile_store import ProfileStore


def _hass_online():
    """A hass whose global online store is pre-seeded (no file I/O)."""
    hass = MagicMock()
    hass.data = {}
    fake = MagicMock()
    fake.async_save = AsyncMock()
    hass.data[store_account._DATA_KEY] = {"store": fake, "data": {"online_enabled": False, "account": {}}}
    return hass


@pytest.mark.asyncio
async def test_global_online_flag_toggles():
    hass = _hass_online()
    assert store_account.online_enabled(hass) is False
    await store_account.async_set_online(hass, True)
    assert store_account.online_enabled(hass) is True


@pytest.mark.asyncio
async def test_global_account_round_trip_and_identity():
    hass = _hass_online()
    await store_account.async_set_account(hass, {"refresh_token": "SECRET", "uid": "u1", "name": "Alice"})
    assert store_account.get_account(hass)["refresh_token"] == "SECRET"
    ident = store_account.get_identity(hass)
    assert ident == {"connected": True, "uid": "u1", "name": "Alice"}
    assert "refresh_token" not in ident
    await store_account.async_clear_account(hass)
    assert store_account.get_account(hass) == {}
    assert store_account.get_identity(hass)["connected"] is False


@pytest.mark.asyncio
async def test_global_account_merges():
    hass = _hass_online()
    await store_account.async_set_account(hass, {"refresh_token": "R", "uid": "u1"})
    await store_account.async_set_account(hass, {"name": "Alice"})
    acct = store_account.get_account(hass)
    assert acct["refresh_token"] == "R" and acct["uid"] == "u1" and acct["name"] == "Alice"


@pytest.fixture
def store():
    hass = MagicMock()

    async def _exec(func, *a, **k):
        return await func(*a, **k) if inspect.iscoroutinefunction(func) else func(*a, **k)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(hass, "e")
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


@pytest.mark.asyncio
async def test_store_account_round_trip(store):
    await store.set_store_account({"uid": "u1", "name": "Alice", "refresh_token": "SECRET", "brand": "Bosch", "model": "WAT"})
    acct = store.get_store_account()
    assert acct["refresh_token"] == "SECRET" and acct["uid"] == "u1"
    ident = store.get_store_identity()
    assert ident == {"connected": True, "uid": "u1", "name": "Alice", "brand": "Bosch", "model": "WAT"}
    assert "refresh_token" not in ident
    await store.clear_store_account()
    assert store.get_store_account() == {}
    assert store.get_store_identity()["connected"] is False


@pytest.mark.asyncio
async def test_set_store_account_merges(store):
    await store.set_store_account({"uid": "u1", "refresh_token": "R", "brand": "Bosch"})
    await store.set_store_account({"model": "WAT28"})  # merge, keep refresh_token/brand
    acct = store.get_store_account()
    assert acct["refresh_token"] == "R" and acct["brand"] == "Bosch" and acct["model"] == "WAT28"


def test_diagnostics_redacts_store_credentials():
    export = {"store_account": {"refresh_token": "SECRET", "id_token": "ID", "uid": "u1", "name": "Alice", "brand": "Bosch"}}
    red = _redact(export)
    sa = red["store_account"]
    assert sa["refresh_token"] == "**REDACTED**"
    assert sa["id_token"] == "**REDACTED**"
    assert sa["uid"] == "**REDACTED**"
    assert sa["name"] == "**REDACTED**"
    assert sa["brand"] == "Bosch"  # non-sensitive context preserved
