# PR8 CI v2 Green-Light Checklist

## ✅ Completed Enhancements

### 1. Bootstrap Robustness
- [x] Minimum CI width enforcement (1.9×SE) for thin samples
- [x] Configurable bootstrap iterations (--n_boot parameter)
- [x] Reproducible random seeds
- [x] Nightly Monte Carlo CI width validation

### 2. Skew Handling
- [x] Automatic skew detection (threshold = 1.0)
- [x] Median bootstrap for skewed returns (|skew| > 1.0)
- [x] Mann-Whitney U test for skewed p-values
- [x] Skewness metrics stored in ci_v2_metadata

### 3. Dependence Handling
- [x] Block bootstrap implementation
- [x] Automatic ACF-based dependence detection
- [x] Block size = horizon_bars
- [x] Circular block bootstrap for edge handling

### 4. FDR Edge Cases
- [x] Adaptive method selection (Holm for n<20, BH for n≥20)
- [x] Validated BH=Holm for n=3
- [x] Monthly grouping with fallback for small samples
- [x] FDR method tracking in metadata

### 5. Sample Size Accuracy
- [x] n_eff based on unique event IDs, not bars
- [x] Event deduplication in queries
- [x] Proper event ID tracking through pipeline

### 6. Power Calculation
- [x] Effect-size based power (Cohen's d approach)
- [x] Non-linear power curve based on lift/SE ratio
- [x] Power score included in all CI calculations

### 7. Performance & Ops
- [x] Chunked processing (configurable chunk size)
- [x] Runtime monitoring (85% threshold warning)
- [x] Job ledger for restart resilience
- [x] DuckDB optimized queries
- [x] Code version hashing for cache invalidation

### 8. Test Coverage
- [x] Block bootstrap coverage test (AR(1) process)
- [x] Skew switch test (log-gamma returns)
- [x] FDR edge case validation
- [x] Trade gate fuzzing at boundaries
- [x] Performance monitoring tests
- [x] Code versioning tests

## 📊 30-Day Side-by-Side Validation Plan

### Week 1-2: Shadow Mode
1. Run CI v2 computation nightly alongside v1
2. Log all decisions to ci_v2_decision_log
3. Compare trade gate decisions daily
4. Monitor computation time and resource usage

### Week 3-4: A/B Testing
1. Enable CI v2 for 10% of requests (rollout_percentage=10)
2. Track key metrics:
   - False positive rate (watch_only that shouldn't be)
   - Lift on allowed trades (median & Sharpe)
   - P95 latency for /council/vote
   - User-reported issues

### Week 5: Full Rollout Prep
1. Increase rollout to 50%
2. Generate daily comparison reports
3. Validate all green-light criteria
4. Prepare rollback plan

## 🚦 Green-Light Criteria

### ✓ Statistical Integrity
- [ ] No increase in false-positive trade rate vs v1
- [ ] Lift on "allowed" trades unchanged or better OOS
- [ ] FDR empirically controlled at 10% ± 2%
- [ ] Bootstrap CI coverage ≥ 93% in validation

### ✓ Performance
- [ ] P95 /council/vote latency < 2× v1 baseline
- [ ] Nightly computation completes in < 85% window
- [ ] No OOM errors in 30-day period
- [ ] Cache hit rate maintained at > 80%

### ✓ Operational
- [ ] All adversarial tests pass in CI
- [ ] Zero data corruption incidents
- [ ] Job restart recovery validated
- [ ] Monitoring alerts configured

### ✓ User Experience
- [ ] /metrics/summary CI uncertainty section functional
- [ ] Rejection reasons clear and actionable
- [ ] No increase in support tickets
- [ ] Documentation updated with plain English explanations

## 📋 Daily Monitoring Checklist

```sql
-- Check CI v2 freshness
SELECT symbol, 
       MAX(computed_at) as last_update,
       CURRENT_TIMESTAMP - MAX(computed_at) as age
FROM consistency_matrix_v2
GROUP BY symbol
HAVING age > INTERVAL '2 days';

-- Compare v1 vs v2 decisions
SELECT 
    v1.trade_gate as v1_gate,
    v2.trade_gate as v2_gate,
    COUNT(*) as count
FROM consistency_matrix v1
JOIN consistency_matrix_v2 v2 
    ON v1.base_pattern = v2.base_pattern
    AND v1.ctx_pattern = v2.ctx_pattern
WHERE v1.computed_at >= CURRENT_DATE - INTERVAL '1 day'
GROUP BY v1.trade_gate, v2.trade_gate;

-- Check FDR control
SELECT fdr_month,
       COUNT(*) as total_tests,
       SUM(CASE WHEN fdr_pass THEN 1 ELSE 0 END) as discoveries,
       AVG(CASE WHEN fdr_pass THEN 1.0 ELSE 0.0 END) as discovery_rate
FROM consistency_matrix_v2
WHERE computed_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY fdr_month
ORDER BY fdr_month DESC;
```

## 🚀 Launch Steps

1. **Pre-launch (Day -7)**
   - [ ] Run full adversarial test suite
   - [ ] Validate job recovery with kill -9 test
   - [ ] Check all monitoring queries work
   - [ ] Review this checklist with team

2. **Soft Launch (Day 0)**
   - [ ] Set rollout_percentage = 10
   - [ ] Enable enhanced logging
   - [ ] Start daily monitoring routine
   - [ ] Create #ci-v2-monitoring Slack channel

3. **Ramp Up (Day 7)**
   - [ ] Review week 1 metrics
   - [ ] Address any issues found
   - [ ] Increase rollout_percentage to 50
   - [ ] Update documentation

4. **Full Launch (Day 30)**
   - [ ] Confirm all green-light criteria met
   - [ ] Set rollout_percentage = 100
   - [ ] Remove v1 code paths
   - [ ] Celebrate with very small emoji 🙂

## 📈 Success Metrics

After 30 days at 100% rollout:
- Trade quality (Sharpe) improved by ≥ 5%
- False discoveries reduced by ≥ 20%
- User trust increased (survey NPS +2)
- Zero statistical integrity incidents

---

**Last Updated**: 2024-01-20
**Version**: 1.0
**Owner**: Trading Analytics Team