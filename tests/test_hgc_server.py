from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import uuid

import numpy as np
import pytest
import torch

from justin_hgc.bin_pose import DEFAULT_BIN_SPEC
from justin_hgc.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointLoadError,
    detect_checkpoint_arm,
    load_checkpoint,
    load_checkpoint_payload,
)
from justin_hgc.protocol import decode_message, encode_message
from justin_hgc.runtime import _farthest_point_backfill, decode_justin_candidates
from justin_hgc.server import (
    HGCGenerationError,
    HGCService,
    dilate_object_mask_6,
    preprocess_depth,
    serve_once,
)
from justin_hgc.templates import canonical_template_q_open


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward_batch(self, batch: dict[str, torch.Tensor]):
        count = batch["point"].shape[1]
        return (
            torch.zeros((1, count, 2, 1), device=batch["point"].device),
            torch.zeros((1, count, 1, 1), device=batch["point"].device),
            torch.zeros((1, count, 12, 1), device=batch["point"].device),
        )


class TinyPaperHead(torch.nn.Module):
    def forward_selected(self, feature: torch.Tensor):
        batch, count, _ = feature.shape
        orientation = torch.zeros((batch, count, 864), device=feature.device)
        orientation[..., 834] = 1.0
        residual = torch.zeros((batch, count, 4), device=feature.device)
        residual[..., 0] = 0.05
        contact = torch.zeros((batch, count, 12), device=feature.device)
        return orientation, residual, contact


class TinyPaperModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.head = TinyPaperHead()

    def forward_grid(self, input_grid: torch.Tensor):
        feature = torch.zeros((1, 2, 64, 64, 64), device=input_grid.device)
        quality = torch.arange(64**3, device=input_grid.device, dtype=torch.float32).reshape(1, 1, 64, 64, 64)
        return feature, quality

    def _select(self, feature_grid: torch.Tensor, indices: torch.Tensor):
        return torch.zeros((1, indices.shape[1], 2), device=feature_grid.device)


class ExpandingPaperHead(torch.nn.Module):
    def forward_selected(self, feature: torch.Tensor):
        batch, count, _ = feature.shape
        flat_index = feature[..., 0]
        highest_100 = flat_index >= float(64**3 - 100)
        orientation = torch.zeros((batch, count, 864), device=feature.device)
        # Bin 474 has palm local-z above the anchor and is rejected by the
        # top-side filter; bin 42 points local-z down and places the palm above.
        orientation[..., 474] = highest_100.to(orientation.dtype)
        orientation[..., 42] = (~highest_100).to(orientation.dtype)
        residual = torch.zeros((batch, count, 4), device=feature.device)
        residual[..., 0] = 0.05
        contact = torch.zeros((batch, count, 12), device=feature.device)
        return orientation, residual, contact


class ExpandingPaperModel(TinyPaperModel):
    def __init__(self) -> None:
        super().__init__()
        self.head = ExpandingPaperHead()

    def _select(self, feature_grid: torch.Tensor, indices: torch.Tensor):
        linear = (
            indices[..., 0] * (64 * 64)
            + indices[..., 1] * 64
            + indices[..., 2]
        )
        return torch.stack((linear, linear), dim=-1).to(dtype=torch.float32)


def _request(**values: object) -> bytes:
    return encode_message(
        {
            "protocol_version": np.asarray(1, dtype=np.int16),
            "op": np.asarray("infer"),
            "depth": np.full((2, 2), 0.2, dtype=np.float32),
            "intrinsics": np.asarray(
                ((100.0, 0.0, 1.0), (0.0, 100.0, 1.0), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            ),
            "T_grid_camera": np.eye(4, dtype=np.float32),
            "seed": np.asarray(7, dtype=np.int64),
            "candidate_count": np.asarray(100, dtype=np.int16),
            **values,
        }
    )


def _fake_candidates(count: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    pose = np.zeros((count, 9), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    q_contact = np.arange(count * 12, dtype=np.float32).reshape(count, 12)
    quality = np.arange(count, 0, -1, dtype=np.float32)
    return (
        {
            "palm_pose": pose,
            "q_contact": q_contact,
            "quality": quality,
            "grasp_point": np.zeros((count, 3), dtype=np.float32),
            "template_index": np.zeros((count,), dtype=np.int64),
        },
        {"pre_top_side": count, "post_top_side": count, "pre_nms": count, "post_nms": count},
    )


def test_preprocess_depth_is_seeded_and_keeps_exact_real_point_rows() -> None:
    depth = np.asarray(
        [[0.2, np.nan, 0.21], [0.22, 0.23, 0.24]], dtype=np.float32
    )
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = intrinsics[1, 1] = 10.0
    first, first_indices, diagnostics = preprocess_depth(
        depth, intrinsics, np.eye(4, dtype=np.float32), seed=4, point_count=10
    )
    second, second_indices, _ = preprocess_depth(
        depth, intrinsics, np.eye(4, dtype=np.float32), seed=4, point_count=10
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_indices, second_indices)
    assert diagnostics["input_valid_count"] == 5
    assert diagnostics["sampled_count"] == 10
    assert np.isfinite(first).all()
    with pytest.raises(HGCGenerationError, match="no valid points"):
        preprocess_depth(
            np.zeros((2, 2), dtype=np.float32),
            intrinsics,
            np.eye(4, dtype=np.float32),
            seed=4,
            point_count=10,
        )


@pytest.mark.parametrize("count", [120, 100, 99, 0])
def test_service_enforces_exact_100_and_stably_truncates(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    monkeypatch.setattr(
        "justin_hgc.server.decode_justin_candidates",
        lambda **_: _fake_candidates(count),
    )
    service = HGCService(TinyModel(), device="cpu")
    response = decode_message(service.handle(_request()))
    if count < 100:
        assert not bool(response["ok"].item())
        assert "exactly 100" in str(response["error_message"].item())
        return
    assert bool(response["ok"].item())
    assert response["T_grid_palm"].shape == (100, 4, 4)
    assert response["q_contact"].shape == (100, 12)
    assert response["quality"].shape == (100,)
    np.testing.assert_array_equal(response["q_contact"][0], np.arange(12, dtype=np.float32))
    assert np.all(response["quality"][:-1] >= response["quality"][1:])


def test_runtime_backfills_nms_suppressed_candidates_without_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 120
    monkeypatch.setattr(
        "justin_hgc.runtime.top_side_mask",
        lambda grasp_point, palm_position: np.ones(count, dtype=bool),
    )
    monkeypatch.setattr(
        "justin_hgc.runtime.aggressive_nms",
        lambda **_: (
            np.arange(74, dtype=np.int64),
            {"pre_nms": count, "post_nms": 74},
        ),
    )
    points = torch.zeros((count, 3), dtype=torch.float32)
    graspable = torch.zeros((count, 2, 1), dtype=torch.float32)
    graspable[:, 1, 0] = torch.linspace(0.1, 2.0, count)
    pose = torch.zeros((count, DEFAULT_BIN_SPEC.channels, 1), dtype=torch.float32)
    joints = torch.arange(count, dtype=torch.float32)[:, None, None].expand(
        -1, 12, 1
    )
    candidates, diagnostics = decode_justin_candidates(
        points=points,
        graspable_logits=graspable,
        pose_logits=pose,
        q_contact=joints,
        minimum_candidates=100,
    )
    assert len(candidates["quality"]) == 100
    assert diagnostics["pre_top_side"] == count
    assert diagnostics["post_top_side"] == count
    assert diagnostics["pre_nms"] == count
    assert diagnostics["post_nms"] == 74
    assert diagnostics["nms_backfill"] == 26
    assert diagnostics["post_backfill"] == 100
    selected_indices = candidates["q_contact"][:, 0].astype(np.int64)
    assert set(range(74)).issubset(set(selected_indices.tolist()))
    assert len(np.unique(selected_indices)) == 100


def test_runtime_fps_backfill_prefers_spatial_separation_over_quality() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.10, 0.0, 0.0]],
        dtype=np.float32,
    )
    selected = _farthest_point_backfill(
        positions=positions,
        quality=np.asarray([0.5, 0.99, 0.1], dtype=np.float32),
        kept=np.asarray([0], dtype=np.int64),
        minimum_candidates=2,
    )
    np.testing.assert_array_equal(selected, np.asarray([0, 2], dtype=np.int64))


def test_runtime_fps_backfill_is_deterministic_and_tie_breaks_quality_then_index() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        dtype=np.float32,
    )
    quality_tie = np.asarray([0.5, 0.2, 0.8], dtype=np.float32)
    first = _farthest_point_backfill(
        positions=positions,
        quality=quality_tie,
        kept=np.asarray([0], dtype=np.int64),
        minimum_candidates=2,
    )
    second = _farthest_point_backfill(
        positions=positions,
        quality=quality_tie,
        kept=np.asarray([0], dtype=np.int64),
        minimum_candidates=2,
    )
    np.testing.assert_array_equal(first, np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(first, second)

    index_tie = _farthest_point_backfill(
        positions=positions,
        quality=np.asarray([0.5, 0.2, 0.2], dtype=np.float32),
        kept=np.asarray([0], dtype=np.int64),
        minimum_candidates=2,
    )
    np.testing.assert_array_equal(index_tie, np.asarray([0, 1], dtype=np.int64))


def test_service_tie_cutoff_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates, counts = _fake_candidates(101)
    candidates["quality"][:] = 0.5
    monkeypatch.setattr(
        "justin_hgc.server.decode_justin_candidates",
        lambda **_: (candidates, counts),
    )
    first = decode_message(HGCService(TinyModel()).handle(_request()))
    second = decode_message(HGCService(TinyModel()).handle(_request()))
    assert first["ok"].item() and second["ok"].item()
    np.testing.assert_array_equal(first["q_contact"], second["q_contact"])
    np.testing.assert_array_equal(first["q_contact"], candidates["q_contact"][:100])


def test_service_rejects_wrong_wire_dtype_before_coercion() -> None:
    service = HGCService(TinyModel(), device="cpu")
    response = decode_message(
        service.handle(_request(depth=np.ones((2, 2), dtype=np.float64)))
    )
    assert not bool(response["ok"].item())
    assert "float32" in str(response["error_message"].item())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"candidate_count": None}, "candidate_count"),
        ({"protocol_version": np.asarray(2, dtype=np.int16)}, "protocol_version"),
        ({"op": np.asarray("unknown")}, "unsupported op"),
        ({"seed": np.asarray(True)}, "seed"),
        ({"intrinsics": np.full((3, 3), np.nan, dtype=np.float32)}, "finite"),
    ],
)
def test_service_rejects_invalid_protocol_fields(
    override: dict[str, object], message: str
) -> None:
    values = {
        "protocol_version": np.asarray(1, dtype=np.int16),
        "op": np.asarray("infer"),
        "depth": np.full((2, 2), 0.2, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
        "T_grid_camera": np.eye(4, dtype=np.float32),
        "seed": np.asarray(7, dtype=np.int64),
        "candidate_count": np.asarray(100, dtype=np.int16),
    }
    for name, value in override.items():
        if value is None:
            values.pop(name)
        else:
            values[name] = value
    response = decode_message(HGCService(TinyModel()).handle(encode_message(values)))
    assert not bool(response["ok"].item())
    assert message in str(response["error_message"].item())


def test_paper_service_rejects_fractional_occupancy() -> None:
    service = HGCService(TinyModel(), arm="paper_modified")
    grid = np.zeros((2, 64, 64, 64), dtype=np.float32)
    grid[0, 0, 0, 0] = 0.75
    response = decode_message(
        service.handle(
            encode_message(
                {
                    "protocol_version": np.asarray(1, dtype=np.int16),
                    "op": np.asarray("infer"),
                    "input_grid": grid,
                    "seed": np.asarray(0, dtype=np.int64),
                    "candidate_count": np.asarray(100, dtype=np.int16),
                }
            )
        )
    )
    assert not bool(response["ok"].item())
    assert "binary 0/1" in str(response["error_message"].item())


def test_paper_service_samples_dense_quality_and_returns_contact_only() -> None:
    grid = np.zeros((2, 64, 64, 64), dtype=np.float32)
    grid[0].reshape(-1)[-200:] = 1.0
    response = decode_message(
        HGCService(TinyPaperModel(), arm="paper_modified").handle(
            encode_message(
                {
                    "protocol_version": np.asarray(1, dtype=np.int16),
                    "op": np.asarray("infer"),
                    "input_grid": grid,
                    "seed": np.asarray(0, dtype=np.int64),
                    "candidate_count": np.asarray(100, dtype=np.int16),
                }
            )
        )
    )
    assert bool(response["ok"].item()), str(response.get("error_message", ""))
    assert response["T_grid_palm"].shape == (100, 4, 4)
    assert response["q_contact"].shape == (100, 12)
    assert "q_open" not in response and "q_squeeze" not in response
    assert np.all(response["quality"][:-1] >= response["quality"][1:])
    assert response["candidate_pool_count"].item() == 100
    assert response["candidate_pool_expansions"].item() == 0


def test_paper_quality_ranking_is_restricted_to_dilated_object_surface() -> None:
    grid = np.zeros((2, 64, 64, 64), dtype=np.float32)
    occupied = np.arange(1000, 1120, dtype=np.int64)
    grid[0].reshape(-1)[occupied] = 1.0
    service = HGCService(TinyPaperModel(), arm="paper_modified")
    model_input, _ = service._prepare_paper({"input_grid": grid})
    _, _, order = service._run_paper_grid(model_input)
    mask = dilate_object_mask_6(torch.from_numpy(grid[0] > 0.5)).reshape(-1)
    self_selected = mask[order.cpu()]
    assert bool(torch.all(self_selected))
    assert len(order) == int(mask.sum())
    assert len(order) > len(occupied)


def test_object_mask_dilation_uses_only_six_face_neighbors() -> None:
    mask = torch.zeros((5, 5, 5), dtype=torch.bool)
    mask[2, 2, 2] = True
    dilated = dilate_object_mask_6(mask)
    assert int(dilated.sum()) == 7
    assert not bool(dilated[1, 1, 1])


def test_paper_service_expands_quality_pool_until_exactly_100_are_valid() -> None:
    grid = np.zeros((2, 64, 64, 64), dtype=np.float32)
    grid[0].reshape(-1)[-200:] = 1.0
    response = decode_message(
        HGCService(ExpandingPaperModel(), arm="paper_modified").handle(
            encode_message(
                {
                    "protocol_version": np.asarray(1, dtype=np.int16),
                    "op": np.asarray("infer"),
                    "input_grid": grid,
                    "seed": np.asarray(0, dtype=np.int64),
                    "candidate_count": np.asarray(100, dtype=np.int16),
                }
            )
        )
    )
    assert bool(response["ok"].item()), str(response.get("error_message", ""))
    assert response["T_grid_palm"].shape == (100, 4, 4)
    assert response["q_contact"].shape == (100, 12)
    assert response["quality"].shape == (100,)
    assert response["candidate_pool_count"].item() == 200
    assert response["candidate_pool_expansions"].item() == 1


def test_service_rejects_decoder_q_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    candidates, counts = _fake_candidates(100)
    candidates["q_contact"] = candidates["q_contact"][:, :11]
    monkeypatch.setattr(
        "justin_hgc.server.decode_justin_candidates",
        lambda **_: (candidates, counts),
    )
    response = decode_message(HGCService(TinyModel()).handle(_request()))
    assert not bool(response["ok"].item())
    assert "q_contact shape" in str(response["error_message"].item())


def test_service_describe_reports_checkpoint_contract() -> None:
    service = HGCService(
        TinyModel(),
        device="cpu",
        checkpoint_path="/tmp/checkpoint.pt",
        checkpoint_sha256="abc",
        checkpoint_epoch=3,
        checkpoint_global_step=12,
        source_commit="unknown",
    )
    response = decode_message(
        service.handle(
            encode_message(
                {
                    "protocol_version": np.asarray(1, dtype=np.int16),
                    "op": np.asarray("describe"),
                }
            )
        )
    )
    assert bool(response["ok"].item())
    assert response["candidate_count"].item() == 100
    assert response["quality_order"].item() == "descending"
    assert tuple(response["output_fields"].tolist()) == ("T_grid_palm", "q_contact", "quality")
    assert tuple(response["template_order"].tolist()) == ("finger2", "finger3", "finger4")
    assert response["checkpoint_sha256"].item() == "abc"


def test_canonical_template_q_open_values_and_order_are_exact() -> None:
    np.testing.assert_allclose(
        canonical_template_q_open(),
        np.asarray(
            (
                (-0.034, 0.0, 0.0, 0.0, 0.415, 0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 1.5707963267948966),
                (-0.194, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.5707963267948966),
                (-0.354, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0, 0.0, 0.4, 0.0),
            ),
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_checkpoint_dispatch_legacy_arm_and_strict_load() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.pt"
        model = TinyModel()
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "model": model.state_dict(),
                "optimizer": {},
                "rng": {},
                "epoch": 2,
                "global_step": 8,
                "config": {},
            },
            path,
        )
        loaded = load_checkpoint(path, model_factory=lambda _: TinyModel())
        assert loaded.metadata.arm == "upstream_faithful"
        assert loaded.metadata.epoch == 2
        assert loaded.metadata.global_step == 8
        with pytest.raises(CheckpointLoadError, match="strict model load"):
            load_checkpoint(path, model_factory=lambda _: torch.nn.Linear(2, 2))


def test_checkpoint_missing_arm_does_not_guess_paper() -> None:
    with pytest.raises(CheckpointLoadError, match="ambiguous"):
        detect_checkpoint_arm(
            {
                "format": CHECKPOINT_FORMAT,
                "config": {"model": {"encoder": "3d_fpn"}},
            }
        )
    with pytest.raises(CheckpointLoadError, match="non-empty config.model"):
        detect_checkpoint_arm(
            {"format": CHECKPOINT_FORMAT, "config": {"arm": "paper_modified"}}
        )


def test_v1_checkpoint_loader_uses_explicit_trusted_pickle_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "trusted.pt"
    path.write_bytes(b"fixture")
    captured: dict[str, object] = {}

    def fake_load(checkpoint: Path, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"format": CHECKPOINT_FORMAT, "config": {}, "model": {}}

    monkeypatch.setattr(torch, "load", fake_load)
    load_checkpoint_payload(path)
    assert captured["weights_only"] is False


def test_in_process_zmq_describe_round_trip() -> None:
    zmq = pytest.importorskip("zmq")

    endpoint = f"inproc://hgc-{uuid.uuid4()}"
    context = zmq.Context.instance()
    server_socket = context.socket(zmq.REP)
    client_socket = context.socket(zmq.REQ)
    server_socket.bind(endpoint)
    client_socket.connect(endpoint)
    worker = threading.Thread(
        target=lambda: serve_once(
            HGCService(TinyModel(), device="cpu"), server_socket
        )
    )
    worker.start()
    try:
        client_socket.send(
            encode_message(
                {
                    "protocol_version": np.asarray(1, dtype=np.int16),
                    "op": np.asarray("describe"),
                }
            )
        )
        described = decode_message(client_socket.recv())
        assert described["ok"].item()
        assert described["candidate_count"].item() == 100
    finally:
        worker.join(timeout=2)
        client_socket.close(linger=0)
        server_socket.close(linger=0)


def test_in_process_zmq_infer_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    zmq = pytest.importorskip("zmq")

    monkeypatch.setattr(
        "justin_hgc.server.decode_justin_candidates",
        lambda **_: _fake_candidates(100),
    )
    endpoint = f"inproc://hgc-infer-{uuid.uuid4()}"
    context = zmq.Context.instance()
    server_socket = context.socket(zmq.REP)
    client_socket = context.socket(zmq.REQ)
    server_socket.bind(endpoint)
    client_socket.connect(endpoint)
    worker = threading.Thread(
        target=lambda: serve_once(
            HGCService(TinyModel(), device="cpu"), server_socket
        )
    )
    worker.start()
    try:
        client_socket.send(_request(seed=np.asarray(3, dtype=np.int64)))
        response = decode_message(client_socket.recv())
        assert response["ok"].item()
        assert response["T_grid_palm"].shape == (100, 4, 4)
    finally:
        worker.join(timeout=2)
        client_socket.close(linger=0)
        server_socket.close(linger=0)
