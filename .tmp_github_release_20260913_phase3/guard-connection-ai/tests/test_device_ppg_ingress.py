from __future__ import annotations

import numpy as np
import pytest

from guard_connection_ai.schemas.device_ppg import DevicePPGPacket
from guard_connection_ai.streaming.device_ppg_ingress import (
    DevicePPGIngressError,
    StatefulCausalADCNormalizer,
    StatefulDevicePPGIngress,
)


def _packet(
    values: np.ndarray,
    *,
    sequence: int = 0,
    start_index: int = 0,
    source_rate: float = 135.0,
) -> DevicePPGPacket:
    end_time_seconds = 1000.0 + (start_index + values.size - 1) / source_rate
    integers = np.rint(values * 10_000).astype(int).tolist()
    return DevicePPGPacket(
        schema_version="1.0",
        stream_id="verity-test",
        sequence_number=sequence,
        configured_sample_rate_hz=source_rate,
        device_timestamp_end_ns=round(end_time_seconds * 1_000_000_000),
        ppg0=integers,
        ppg1=[value + 2 for value in integers],
        ppg2=[value - 2 for value in integers],
        ambient0=[500] * values.size,
    )


def test_four_channels_are_validated_and_ambient_is_not_subtracted() -> None:
    packet = _packet(np.asarray([1.0, 2.0, 3.0]))
    ingress = StatefulDevicePPGIngress(target_sample_rate_hz=135)
    result = ingress.process(packet)

    np.testing.assert_allclose(result.ppg, [10_000, 20_000, 30_000])
    assert result.source_packet.ambient0 == [500, 500, 500]
    assert np.all(result.ppg_valid)


def test_demo_sources_require_scenario_and_live_input_cannot_claim_one() -> None:
    packet = _packet(np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="demo_scenario_id"):
        DevicePPGPacket(
            **{
                **packet.__dict__,
                "source_mode": "recorded_demo_replay",
            }
        )
    with pytest.raises(ValueError, match="live device"):
        DevicePPGPacket(
            **{
                **packet.__dict__,
                "demo_scenario_id": "not-live",
            }
        )
    with pytest.raises(ValueError, match="demo_scenario_id"):
        DevicePPGPacket(
            **{
                **packet.__dict__,
                "source_mode": "synthetic_demo",
            }
        )


def test_135_to_125_resampling_is_packet_boundary_invariant() -> None:
    values = np.sin(2 * np.pi * 1.2 * np.arange(270) / 135)
    whole = StatefulDevicePPGIngress(target_sample_rate_hz=125).process(_packet(values))
    chunked = StatefulDevicePPGIngress(target_sample_rate_hz=125)
    first = chunked.process(_packet(values[:103]))
    second = chunked.process(_packet(values[103:], sequence=1, start_index=103))

    np.testing.assert_allclose(
        np.concatenate((first.ppg, second.ppg)), whole.ppg, rtol=0, atol=1e-5
    )
    np.testing.assert_array_equal(
        np.concatenate((first.ppg_valid, second.ppg_valid)), whole.ppg_valid
    )


def test_missing_or_saturated_channels_produce_explicit_invalid_samples() -> None:
    packet = _packet(np.asarray([1.0, 2.0, 3.0]))
    packet = DevicePPGPacket(
        **{
            **packet.__dict__,
            "ppg0": [10_000, None, 30_000],
            "ppg1": [10_002, None, 30_002],
            "saturation": {"ppg2": [False, True, False]},
        }
    )
    result = StatefulDevicePPGIngress(target_sample_rate_hz=135).process(packet)

    assert result.ppg_valid.tolist() == [True, False, True]
    assert np.isnan(result.ppg[1])


@pytest.mark.parametrize(
    ("accepted_packets", "sequence", "code"),
    ((1, 0, "duplicate_packet"), (2, 0, "out_of_order"), (1, 2, "sequence_gap")),
)
def test_packet_order_errors_do_not_advance_state(
    accepted_packets: int, sequence: int, code: str
) -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    ingress = StatefulDevicePPGIngress(target_sample_rate_hz=135)
    for accepted_sequence in range(accepted_packets):
        ingress.process(
            _packet(
                values,
                sequence=accepted_sequence,
                start_index=accepted_sequence * values.size,
            )
        )
    with pytest.raises(DevicePPGIngressError) as error:
        ingress.process(
            _packet(values, sequence=sequence, start_index=accepted_packets * values.size)
        )
    assert error.value.code == code
    assert error.value.expected_sequence_number == accepted_packets


def test_new_stream_requires_explicit_reset() -> None:
    ingress = StatefulDevicePPGIngress(target_sample_rate_hz=135)
    packet = _packet(np.asarray([1.0, 2.0]))
    ingress.process(packet)
    other = DevicePPGPacket(**{**packet.__dict__, "stream_id": "other", "sequence_number": 1})
    with pytest.raises(DevicePPGIngressError) as error:
        ingress.process(other)
    assert error.value.code == "stream_changed"


def test_sub_target_period_packet_can_emit_empty_chunk_without_losing_state() -> None:
    ingress = StatefulDevicePPGIngress(target_sample_rate_hz=125)
    first = ingress.process(_packet(np.asarray([1.0])))
    second = ingress.process(
        _packet(np.asarray([2.0]), sequence=1, start_index=1)
    )
    third = ingress.process(
        _packet(np.asarray([3.0]), sequence=2, start_index=2)
    )

    assert first.ppg.size == 1
    assert second.ppg.size == 0
    assert third.ppg.size == 1


def test_adc_normalizer_is_causal_chunk_invariant_and_preserves_missingness() -> None:
    values = 500_000 + 20_000 * np.sin(2 * np.pi * np.arange(270) / 135)
    values[100] = np.nan
    batch_normalizer = StatefulCausalADCNormalizer(sample_rate_hz=135)
    chunk_normalizer = StatefulCausalADCNormalizer(sample_rate_hz=135)

    batch = batch_normalizer.process(values)
    chunked = np.concatenate(
        [chunk_normalizer.process(values[:80]), chunk_normalizer.process(values[80:])]
    )

    np.testing.assert_allclose(batch, chunked, equal_nan=True)
    assert np.isnan(batch[100])
    assert np.nanmin(batch) >= -0.1 - 1e-12
    assert np.nanmax(batch) <= 1.1 + 1e-12
