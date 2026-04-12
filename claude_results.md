# Phase 23 Track A v2: Layout-aware, Scenario-agnostic

## v1 실패 교훈
- Scenario aggregate → train/test 분포 차이로 Public 10.28 (baseline 9.86 대비 악화)
- Test layout 50/100 unseen → scenario aggregate 무의미
- GroupKFold CV 8.79 → Public 10.28 (gap 1.49)

## v2 설계 원칙
1. Scenario aggregate 완전 제거
2. Row-level features + Layout-aware features만
3. Unseen layout 대응 (구조 변수 x 운영 변수 interaction)

## 9-Step Pipeline
1. **Saturation**: pack/robot/dock margin, cascading trigger, inflow pressure (~13)
2. **Queueing W**: rho/(1-rho), Little's Law, bottleneck (~7)
3. **Layout Capacity**: density, congestion potential, oneway penalty (~7)
4. **Layout x Operation Interaction**: aisle x traffic, intersection x congestion, charger demand (~12)
5. **Within-Layout Percentile**: 11 key cols rank(pct=True) by layout_id (train+test unsupervised)
6. **Layout Type**: one-hot + KFold target encoding
7. **Feature Selection**: 3-fold GroupKFold importance, zero ALL folds + bottom 5% removal
8. **5-Fold CV**: LGB Huber 5000 trees, GroupKFold by layout_id (GPU + CPU fallback)
9. **Save**: submission, OOF, checkpoint
