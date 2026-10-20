"""Linear models and statistics of the order-flow baselines."""

from dof.lob.stats.linear import (
    SuffStats,
    adj_r2,
    block_t_test,
    ols_small,
    r2_os,
    r2_vs_eval_mean,
    r2_zero,
    solve_ridge,
    suff_stats_init,
    suff_stats_update,
)

__all__ = ["SuffStats", "adj_r2", "block_t_test", "ols_small", "r2_os", "r2_vs_eval_mean", "r2_zero", "solve_ridge",
           "suff_stats_init", "suff_stats_update"]
