# mySPiKE

SPiKE 기반 다중 도메인 추천 시스템 (알고리즘 파트).
원본 SPiKE 코드는 참고용으로만 쓰고, 이 저장소에서 도메인 일반화 파이프라인을 구현한다.

## 구조

```
mySPiKE/
├── validate_dataset.py     input validation for data team submissions   (Step 2)
├── preprocess.py           raw/ -> prep/<version>/                       (Step 2)
├── make_profiles.py        prep/<version>/ -> profile/<version>_<method>/ (Step 3)
├── configs/profile/        one JSON file per profile method, e.g. qwen3-4b.json
├── tools/                  one-off helpers (e.g. author book data -> spec format)
├── tests/                  pytest
├── docs/                   design documents (PDF)
└── data/                   project data, not tracked by git
    └── <domain>/
        ├── raw/            data team submission (5 files)
        ├── prep/<ver>/     preprocessing output
        └── profile/<ver>_<method>/   profiles + embeddings (Step 3)
            ├── raw/        item_raw_profiles.jsonl ...
            ├── meta.json   prep hash, per-stage config / inputs / stats
            └── profile.log
```

## 환경

- Python 3.10 이상. `validate_dataset.py`, `preprocess.py`와 `make_profiles.py`의 원시 프로필 단계는 **표준 라이브러리만** 사용한다 (별도 설치 없이 실행 가능).
- 테스트: 가상환경을 만들고 `pip install -r requirements.txt` 후 `pytest`
  ```bash
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m pytest
  ```

## 사용법

```bash
# 제출 데이터 검증 (에러가 있으면 종료 코드 1)
python validate_dataset.py data/<domain>/raw [--json report.json]

# 전처리 (내부에서 검증을 다시 실행, 에러가 있으면 중단)
python preprocess.py data/<domain> --version v1

# 프로필 생성 (현재 구현된 단계: item_raw)
python make_profiles.py data/<domain> --prep v1 --config configs/profile/qwen3-4b.json --stage item_raw
#   출력: data/<domain>/profile/v1_qwen3-4b/  (폴더 이름 = <prep 버전>_<설정 파일 이름>)
#   종료 코드: 0 성공 / 2 설정 오류 / 3 prep·추적 오류
#   완료된 단계는 건너뛴다. 다시 만들려면 출력 파일을 직접 지운다.
#   설정을 바꾸려면 새 이름의 설정 파일을 만든다 (새 폴더가 생김).

# 저자 book 데이터 -> 명세 형식 5개 파일 (data/book/raw/)
python tools/convert_book.py ../SPiKE/spike/datasets/kg/amazon-book_kg data/book/raw

# 일부러 참고 조건을 어긴 시험용 데이터 (data/book_noisy/raw/ + noisy_manifest.json)
python tools/make_noisy_book.py data/book/raw data/book_noisy
```

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| SPiKE_데이터_명세서 | 구축팀 제출 형식 (raw/ 5개 파일) |
| SPiKE_데이터셋_구조_보고서 | 저자 배포 데이터 분석 |
| SPiKE_데이터_전처리_설계서 | Step 2 설계 |
| SPiKE_데이터_디렉토리_규약 | data/ 폴더 구조 |
| SPiKE_데이터_전처리_구현_결과_보고서 | 검증·전처리 구현과 확인 결과 |
| SPiKE_LLM_프로필_설계서 | Step 3 원시·LLM 프로필 설계 |
| SPiKE_개선_사항 | 추후 과제, 알려진 한계, 구축팀 데이터 개선 요청 (계속 갱신) |
