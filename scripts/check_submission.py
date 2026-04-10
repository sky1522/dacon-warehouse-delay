"""Submission 파일 검증."""
import pandas as pd
import numpy as np
import sys

def check(path):
    sample = pd.read_csv('data/sample_submission.csv')
    sub = pd.read_csv(path)

    print(f"=== Checking: {path} ===")

    # ID 일치
    assert list(sub.columns) == list(sample.columns), f"Column mismatch: {sub.columns} vs {sample.columns}"
    assert (sub['ID'].values == sample['ID'].values).all(), "ID order mismatch"
    print(f"  ID: OK ({len(sub)} rows)")

    # NaN
    n_nan = sub['avg_delay_minutes_next_30m'].isna().sum()
    assert n_nan == 0, f"NaN found: {n_nan}"
    print(f"  NaN: OK (0)")

    # Range
    pred = sub['avg_delay_minutes_next_30m']
    print(f"  Range: [{pred.min():.2f}, {pred.max():.2f}]")
    print(f"  Mean: {pred.mean():.2f}, Median: {pred.median():.2f}")
    assert (pred >= 0).all(), "Negative predictions found"
    print(f"  Non-negative: OK")

    # Distribution
    print(f"  Quantiles: 25%={pred.quantile(0.25):.2f}, 50%={pred.quantile(0.5):.2f}, "
          f"75%={pred.quantile(0.75):.2f}, 95%={pred.quantile(0.95):.2f}, 99%={pred.quantile(0.99):.2f}")

    print(f"  PASSED")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'output/submission_phase16.csv'
    check(path)
