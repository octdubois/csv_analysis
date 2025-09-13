from csvi.bench import run_benchmark


def test_bench_small() -> None:
    res = run_benchmark(100_000)
    assert res["rows"] == 100_000
    assert res["resample_s"] >= 0.0
