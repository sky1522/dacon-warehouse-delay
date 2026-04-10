# Phase Runner

Phase 스크립트 실행 및 seed 변경 패턴.

## Seed 변경
```bash
# sed로 seed 변경 (random_state, ckpt 이름, submission 이름)
sed -i 's/random_state=42/random_state=777/g' run_phaseNN.py
sed -i 's/ckpt_phaseNN_/ckpt_phaseNNs777_/g' run_phaseNN.py
sed -i 's/submission_phaseNN/submission_phaseNNs777/g' run_phaseNN.py
```

## GPU/CPU 전환
- LGBMRegressor: `device='gpu'` 추가 가능 (Colab T4)
- LGBMClassifier: CPU only (class_weight 호환 문제)
- XGBoost: `tree_method='hist'` (GPU 시 `device='cuda'`)
- CatBoost: `task_type='GPU'` 가능

## Drive 백업
- 각 모델 학습 완료 직후 즉시 `shutil.copy` → Drive
- 마지막에 몰아서 하지 않음 (Colab 중단 대비)

## Colab 실행 패턴
```python
# 셀 1: Drive mount + requirements
from google.colab import drive
drive.mount('/content/drive')
!pip install -q pytorch-tabnet

# 셀 2: 코드 복사 + 실행
!cp /content/drive/MyDrive/dacon_ckpt/run_phaseNN.py .
!python run_phaseNN.py
```
