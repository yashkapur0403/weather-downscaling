"""The advisory must be based on the Layer-1 rainfall for THIS Panchayat and date.

Before this, GET /api/advisory took `rainfall_mm`, `date` and `panchayat_name`
from the caller, used `panchayat_id` only to look up auxiliary data, and echoed
the client's date back as `data_date`. A stale client value (or the frontend's
falsy-zero fallback) could therefore produce confident advice about a different
Panchayat or day with nothing in the trace to show for it.

These tests pin the replacement behaviour: resolve, check, refuse.
"""
from __future__ import annotations

import pytest

from conftest import (DATE, ID_ALERT, ID_INFO, ID_MASKED, ID_TINY, ID_UNKNOWN,
                      ID_WARNING, ID_ZERO)


@pytest.fixture
def client(app_with_data):
    from fastapi.testclient import TestClient
    return TestClient(app_with_data)


def adv(client, **q):
    base = dict(panchayat_id=ID_ALERT, crop="rice", stage="flowering",
                date=DATE, panchayat_name="GP11")
    base.update(q)
    return client.get("/api/advisory", params=base)


# --------------------------------------------------------------- happy paths
def test_supplied_rainfall_matching_the_field_is_verified(client):
    r = adv(client, rainfall_mm=70.0)
    assert r.status_code == 200
    b = r.json()
    v = b["verification"]
    assert v["rainfall"] == "verified_against_layer1"
    assert v["expected_mm"] == 70.0 and v["supplied_mm"] == 70.0
    assert v["panchayat_id"] == ID_ALERT and v["date"] == DATE
    assert v["location_precision"] == "exact"
    assert v["source"] and v["cell"] is not None
    # the rules ran on the stored value, not on the caller's string
    assert b["evidence"]["rainfall_mm"] == 70.0
    assert b["severity"] == "alert"


def test_rainfall_can_be_omitted_and_is_resolved_server_side(client):
    r = adv(client)                      # no rainfall_mm at all
    assert r.status_code == 200
    b = r.json()
    assert b["verification"]["rainfall"] == "resolved_from_layer1"
    assert b["evidence"]["rainfall_mm"] == 70.0
    assert b["severity"] == "alert"


def test_rounding_tolerance_accepts_a_faithful_client(client):
    """Values are rounded to 2 dp at the boundary; half a step must not 409."""
    assert adv(client, rainfall_mm=70.005).status_code == 200
    assert adv(client, rainfall_mm=69.995).status_code == 200


def test_zero_rainfall_is_a_real_answer_not_a_missing_one(client):
    """0.0 mm must survive as 0.0 - the same falsiness that broke the frontend."""
    r = adv(client, panchayat_id=ID_ZERO, rainfall_mm=0.0)
    assert r.status_code == 200
    b = r.json()
    assert b["evidence"]["rainfall_mm"] == 0.0
    assert b["verification"]["rainfall"] == "verified_against_layer1"


def test_each_panchayat_is_checked_against_its_own_value(client):
    # every id resolves against its OWN stored value, and the rules then run
    assert adv(client, panchayat_id=ID_WARNING, rainfall_mm=30.0).json()["severity"] == "warning"
    assert adv(client, panchayat_id=ID_INFO, rainfall_mm=12.0).json()["severity"] == "info"
    tiny = adv(client, panchayat_id=ID_TINY, rainfall_mm=2.0).json()
    assert tiny["verification"]["expected_mm"] == 2.0
    assert tiny["severity"] == "watch"          # R2_IRRIGATION (low) below 3.0 mm
    assert tiny["fired_rules"] and tiny["fired_rules"][0]["rule_id"] == "R2_IRRIGATION"


# ------------------------------------------------------------------ refusals
def test_mismatched_rainfall_is_rejected(client):
    r = adv(client, rainfall_mm=999.0)
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["supplied_mm"] == 999.0 and d["expected_mm"] == 70.0
    assert d["panchayat_id"] == ID_ALERT and d["date"] == DATE
    assert "does not match" in d["error"]


def test_a_stale_value_from_another_panchayat_is_rejected(client):
    """The exact failure the frontend's falsy-zero fallback could produce."""
    stale = 30.0                                  # GP12's value, sent for GP11
    r = adv(client, panchayat_id=ID_ALERT, rainfall_mm=stale)
    assert r.status_code == 409
    assert r.json()["detail"]["expected_mm"] == 70.0


def test_date_outside_the_served_grid_is_refused(client):
    r = adv(client, rainfall_mm=70.0, date="2021-07-10")
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "date_unavailable"


def test_masked_cell_is_refused_rather_than_guessed(client):
    r = adv(client, panchayat_id=ID_MASKED, rainfall_mm=21.0)
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "masked_cell"


def test_unknown_panchayat_is_refused(client):
    r = adv(client, panchayat_id=ID_UNKNOWN, rainfall_mm=21.0)
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "unknown_panchayat"


def test_refusal_does_not_leak_a_severity(client):
    """A refused request must not return a half-built advisory."""
    b = adv(client, rainfall_mm=999.0).json()
    assert "severity" not in b and "actions" not in b


# ------------------------------------------------- no data layer (text-only)
def test_without_a_data_source_rainfall_is_required(app_without_data):
    from fastapi.testclient import TestClient
    c = TestClient(app_without_data)
    r = c.get("/api/advisory", params={"panchayat_id": 1})
    assert r.status_code == 422


def test_without_a_data_source_the_answer_is_labelled_unverified(app_without_data):
    from fastapi.testclient import TestClient
    c = TestClient(app_without_data)
    r = c.get("/api/advisory", params={"panchayat_id": 1, "rainfall_mm": 70.0})
    assert r.status_code == 200
    assert r.json()["verification"]["rainfall"] == "unverified_no_data_source"
