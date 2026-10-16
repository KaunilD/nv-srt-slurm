# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client imports retain normalized requests, not complete decoded exports."""

import json
import tracemalloc

import pytest

from srtctl.dsight.importer import Importer

ORIGIN = 1789642711000000000


def record(index, *, phase="profiling"):
    metadata = {
        "request_id": f"request-{index}",
        "request_start_ns": ORIGIN + index * 1_000_000_000,
        "request_end_ns": ORIGIN + (index + 1) * 1_000_000_000,
    }
    if phase is not None:
        metadata["benchmark_phase"] = phase
    return {"metadata": metadata, "metrics": {"time_to_first_token": {"value": 100}}}


def test_large_unused_payloads_do_not_accumulate(tmp_path):
    source = tmp_path / "profile_export.jsonl"
    with source.open("w") as stream:
        for index in range(300):
            row = record(index)
            row["unused_payload"] = "x" * 65536
            stream.write(json.dumps(row) + "\n")
    importer = Importer(tmp_path, client=source)
    tracemalloc.start()
    try:
        importer.clients()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(importer.requests) == 300
    assert importer.requests[-1]["id"] == "request-299"
    assert importer.requests[-1]["ttft_ms"] == 100
    # The previous list of decoded rows alone retains over 18 MiB here.
    assert peak < 6 * 1024 * 1024


def test_two_pass_import_preserves_origin_line_evidence_and_audit(tmp_path):
    source = tmp_path / "profile_export.jsonl"
    rows = [record(10), record(-5, phase="warmup"), record(0, phase=None)]
    source.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n")
    importer = Importer(tmp_path, client=source)
    importer.clients()
    assert importer.origin == ORIGIN
    assert importer.duration == 11
    assert [r["id"] for r in importer.requests] == ["request-0", "request-10"]
    assert [r["evidence"][1] for r in importer.requests] == [6, 2]
    assert importer.audit["client_requests"] == 2
    assert importer.audit["excluded_client_rows"] == 1
    assert importer.audit["clients_without_phase"] == 1


def test_streaming_import_preserves_invalid_record_line(tmp_path):
    source = tmp_path / "profile_export.jsonl"
    source.write_text(json.dumps(record(0)) + "\n\n{invalid json}\n")
    with pytest.raises(ValueError, match=r"profile_export.jsonl:3: invalid client record"):
        Importer(tmp_path, client=source).clients()
