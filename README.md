# mySPiKE

SPiKE 기반 다중 도메인 추천 시스템 (알고리즘 파트).
원본 SPiKE 코드는 참고용으로만 쓰고, 이 저장소에서 도메인 일반화 파이프라인을 구현한다.

## 구조

```
mySPiKE/
├── validate_dataset.py     input validation for data team submissions   (Step 2)
├── preprocess.py           raw/ -> prep/<version>/                       (Step 2)
├── tools/                  one-off helpers (e.g. author book data -> spec format)
├── tests/                  pytest
├── docs/                   design documents (PDF)
└── data/                   project data, not tracked by git
    └── <domain>/
        ├── raw/            data team submission (5 files)
        ├── prep/<ver>/     preprocessing output
        └── profile/<ver>/  profiles + embeddings (Step 3)
```

## 환경

- Python 3.10 이상. `validate_dataset.py`, `preprocess.py`는 **표준 라이브러리만** 사용한다 (구축팀이 별도 설치 없이 실행 가능).
- 테스트: `pip install -r requirements.txt` 후 `pytest`

## 사용법

```bash
# 저자 book 데이터 -> 명세 형식 5개 파일 (data/book/raw/)
python tools/convert_book.py ../SPiKE/spike/datasets/kg/amazon-book_kg data/book/raw
```

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| SPiKE_데이터_명세서 | 구축팀 제출 형식 (raw/ 5개 파일) |
| SPiKE_데이터셋_구조_보고서 | 저자 배포 데이터 분석 |
| SPiKE_데이터_전처리_설계서 | Step 2 설계 |
| SPiKE_데이터_디렉토리_규약 | data/ 폴더 구조 |
