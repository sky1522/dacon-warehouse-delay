codex --reasoning high "C:\dev\dacon-warehouse-delay 프로젝트 전수조사. codex_results.md에 저장.

## 목표
파일 정리, 문서 현황, 불필요한 파일 식별을 위한 종합 조사

## 1. 루트 레벨 파일 전수조사
모든 .py, .md, .csv, .json, .yaml 파일 나열 후 각각:
- 파일 크기, 마지막 수정 시간
- 목적 (코드 첫 20줄 또는 주석 기반 추정)
- 상태: [active/deprecated/duplicate/temporary/unknown]
- 삭제/유지/리팩터 판단
- 중복 파일 그룹 식별 (blend_temp.py, codex_results.new.md 등)

## 2. run_phase*.py 의존성 맵
각 Phase 파일에 대해:
- 입력: 어떤 ckpt_*.pkl 또는 CSV 파일을 로드하는가
- 출력: 어떤 ckpt_*.pkl, submission_*.csv 생성하는가
- 다른 Phase에 의존하는지 (예: phase18 → phase16 ckpt)
- 고아 파일 식별 (아무도 참조 안 하고 결과도 안 쓰이는 파일)
- 삭제 가능 후보 표시

## 3. 문서 파일 현황
각 .md 파일 분석:
- CLAUDE.md, README.md, PROGRESS.md, DECISION.md, CHANGELOG.md 존재 여부
- claude_results.md: 어느 Phase까지 기록? 빠진 Phase?
- codex_results.md, codex_results.old.md, codex_results.new.md: 중복/백업 상태
- claude_prompt.md, codex_prompt.md: 어떤 내용? 유지 가치?
- docs/phases/*.md: 모든 Phase 커버하는지, 내용 완성도
- docs/decisions/*.md: 있는지
- docs/analysis/*.md: 있는지

## 4. 체크포인트/산출물 파일 현황
- output/ 실제 존재하는 ckpt_*.pkl 목록 (로컬)
- claude_results.md가 언급하는 ckpt vs 실제 존재 ckpt 대조
- submission_*.csv 중복/버전 (submission_phase16.csv vs submission_phase16_backup.csv 등)
- 삭제 가능 임시 파일 (blend_p7p8_60.csv, blend_temp.py 등)

## 5. 정리 권장 사항 (카테고리별)
### A. 즉시 삭제 가능
- 임시 파일, 중복 백업, 미완성 실험
### B. 아카이브 (archive/ 폴더로 이동)
- Phase 1~10 등 구식이지만 참조 가치 있는 것
### C. 리팩터 필요
- 코드 중복 심한 phase15/16/17/18 → 공통 모듈 분리
### D. 문서화 필요
- README.md, PROGRESS.md, DECISION.md 생성
### E. 업데이트 필요
- 미완성 섹션, 오래된 정보

## 6. 우선순위 정리 계획
- P0 (오늘): 필수 문서 생성 (README, PROGRESS, DECISION)
- P1 (내일): 불필요 파일 삭제, archive 이동
- P2 (이번 주): 코드 리팩터
- P3 (다음 주): 고급 문서화

결과는 마크다운, 각 섹션 H2 헤더, 파일은 표 형식으로 나열"