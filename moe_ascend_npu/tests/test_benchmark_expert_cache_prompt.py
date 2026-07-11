"""CPU-only tests for fixed-prompt expert-cache benchmark parsing."""

from pathlib import Path

import pytest

from benchmark_expert_cache_prompt import enrich_results, parse_server_log, summarize


def test_parse_server_log_extracts_cache_metrics(tmp_path: Path):
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            [
                "[ExpertCache] allocated active=256 spare=8 slot=2.53 MiB total=0.65 GiB",
                "[ExpertCache] step=656 active=256 window_hit=39.19% swaps=256",
                "[ExpertCache] step=784 active=256 window_hit=37.48% swaps=256",
                "[ExpertCache] steady update interval backed off to 256 steps",
                "[ExpertCache] step=1040 active=256 window_hit=38.66% swaps=256",
                "[ExpertCache] step=1296 active=256 window_hit=38.81% swaps=256",
                "[ExpertCache] steady update interval backed off to 512 steps",
                "Decode batch, gen throughput (token/s): 43.93, #queue-req: 0",
                "Decode batch, gen throughput (token/s): 44.05, #queue-req: 0",
                "Decode batch, gen throughput (token/s): 43.21, #queue-req: 0",
            ]
        ),
        encoding="utf-8",
    )

    stats = parse_server_log(log_path, hit_tail=4)

    assert stats["allocation"] == {
        "active": 256,
        "spare": 8,
        "slot_mib": 2.53,
        "total_gib": 0.65,
    }
    assert stats["hit_rate_samples"] == pytest.approx([0.3919, 0.3748, 0.3866, 0.3881])
    assert stats["hit_rate_last"] == pytest.approx(0.3881)
    assert stats["hit_rate_median_tail"] == pytest.approx((0.3866 + 0.3881) / 2)
    assert stats["swap_last"] == 256
    assert stats["observed_backoff_steps"] == [256, 512]
    assert stats["local_decode_tps_samples"] == [43.93, 44.05, 43.21]
    assert stats["local_decode_tps_median_tail16"] is None


def test_summarize_drops_first_request_and_tracks_hash_stability():
    rows = [
        {"latency_s": 10.0, "sha256_12": "abc"},
        {"latency_s": 8.0, "sha256_12": "abc"},
        {"latency_s": 7.0, "sha256_12": "abc"},
        {"latency_s": 9.0, "sha256_12": "abc"},
    ]

    summary = summarize(
        mode="cache",
        cache_size=256,
        output_tokens=320,
        request_rows=rows,
        log_stats={"hit_rate_last": 0.38},
    )

    assert summary["mode"] == "cache"
    assert summary["k"] == 256
    assert summary["latency_s_samples"] == [10.0, 8.0, 7.0, 9.0]
    assert summary["latency_s_median_drop_first"] == 8.0
    assert summary["throughput_tps_median_drop_first"] == 40.0
    assert summary["output_hashes"] == ["abc"]
    assert summary["output_hash_stable"] is True
    assert summary["log"] == {"hit_rate_last": 0.38}


def test_summarize_marks_hash_drift():
    rows = [
        {"latency_s": 8.0, "sha256_12": "abc"},
        {"latency_s": 8.0, "sha256_12": "def"},
    ]

    summary = summarize(
        mode="cpu_q4_0",
        cache_size=None,
        output_tokens=320,
        request_rows=rows,
        log_stats={},
    )

    assert summary["output_hashes"] == ["abc", "def"]
    assert summary["output_hash_stable"] is False


def test_enrich_results_adds_speedups_and_cache_footprint():
    results = [
        {
            "mode": "cpu_q4_0",
            "k": None,
            "latency_s_median_drop_first": 8.0,
            "throughput_tps_median_drop_first": 40.0,
            "output_hashes": ["same"],
            "log": {"local_decode_tps_median_tail16": 40.0},
        },
        {
            "mode": "cache",
            "k": 256,
            "latency_s_median_drop_first": 7.5,
            "throughput_tps_median_drop_first": 42.0,
            "output_hashes": ["same"],
            "log": {
                "local_decode_tps_median_tail16": 44.0,
                "allocation": {"active": 256, "spare": 8},
            },
        },
    ]

    derived = enrich_results(
        results,
        output_tokens=320,
        slot_mib=2.53125,
        num_moe_layers=48,
        num_experts=128,
    )

    assert derived["exact_text_hash_match_across_modes"] is True
    assert derived["reference_output_hashes"] == ["same"]
    assert derived["cpu_latency_s_median_drop_first"] == 8.0
    assert derived["cpu_local_decode_tps_median_tail16"] == 40.0

    cache = results[1]
    assert cache["throughput_tps_median_drop_first"] == 42.667
    assert cache["speedup_vs_cpu_e2e"] == 1.0667
    assert cache["speedup_vs_cpu_local_decode"] == 1.1
    assert cache["expert_instance_fraction"] == 0.0416666667
    assert cache["physical_cache_gib"] == pytest.approx(0.6525878906)


def test_enrich_results_detects_cross_mode_hash_mismatch():
    results = [
        {
            "mode": "cpu_q4_0",
            "k": None,
            "latency_s_median_drop_first": 8.0,
            "output_hashes": ["cpu"],
            "log": {},
        },
        {
            "mode": "cache",
            "k": 256,
            "latency_s_median_drop_first": 8.0,
            "output_hashes": ["cache"],
            "log": {},
        },
    ]

    derived = enrich_results(
        results,
        output_tokens=320,
        slot_mib=2.53125,
        num_moe_layers=48,
        num_experts=128,
    )

    assert derived["exact_text_hash_match_across_modes"] is False
