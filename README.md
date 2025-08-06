# DDPM 프로젝트 모음

이 레포지토리는 DDPM(Denoising Diffusion Probabilistic Models) 관련 실험ㆍ재현 코드 및 스크립트를 모아둔 저장소입니다. 학습/추론을 위한 핵심 코드만 선별하여 포함하고 있으며, 대용량 데이터·캐시·가상환경 등은 저장소에 포함하지 않습니다.

## 주요 디렉터리 및 파일

| 경로 | 설명 |
|------|------|
| `baselines-reproduction/` | DDPM 및 기타 diffusion 모델 baseline 재현 코드 |
| `ot_sf/` | Optimal Transport–Score Function(OT-SF) 관련 실험 코드와 증명 |
| `ddpm_setup_complete.sh` | 환경 구축 자동화 스크립트 │
| `ddpm_manual_steps.md` | 수동 설정 및 추가 설명 문서 |
| `.gitignore` | 불필요한 파일(데이터·캐시·가상환경 등) 제외 설정 |
| `requirements.txt` (선택) | 파이썬 의존성 목록 |

## 설치 방법

```bash
# 1. 저장소 클론
$ git clone https://github.com/junhoso1219/ddpm.git
$ cd ddpm

# 2. (선택) 가상환경 생성
$ python -m venv .venv
$ source .venv/bin/activate

# 3. 의존성 설치
$ pip install -r requirements.txt  # requirements.txt가 있을 경우

# 4. 추가 스크립트 실행(선택)
$ bash ddpm_setup_complete.sh
```

## 사용 예시

```bash
# 베이스라인 학습 예시
python baselines-reproduction/train.py --config configs/base.yml
```

더 자세한 내용은 각 디렉터리의 README 또는 코드 주석을 참고해 주세요.

## 라이선스

MIT License (파일 최상단 라이선스 내용을 참조)