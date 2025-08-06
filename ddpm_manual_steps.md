# DDPM 베이스라인 훈련 - 수동 실행 가이드

## 전제조건
```bash
# venv 가상환경이 미리 생성되어 있어야 함
cd ~/projects/baselines-reproduction
source .venv/bin/activate
```

## 1. diffusers 라이브러리 클론
```bash
git clone https://github.com/huggingface/diffusers.git
```

## 2. 필수 패키지 설치
```bash
pip install accelerate transformers datasets torchvision tensorboard
```

## 3. diffusers 개발 버전 설치 (⚠️ 중요)
```bash
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git
```

## 4. accelerate 설정 파일 생성
```bash
mkdir -p ~/.cache/huggingface/accelerate

cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
```

## 5. CIFAR-10 컬럼명 문제 수정 (⚠️ 핵심)
```bash
sed -i 's/examples\["image"\]/examples["img"]/g' diffusers/examples/unconditional_image_generation/train_unconditional.py
```

## 6. 훈련 결과 폴더 생성
```bash
mkdir -p runs/ddpm-cifar10-repro
```

## 7. DDPM 훈련 실행
```bash
accelerate launch \
    --mixed_precision="fp16" \
    --num_processes=1 \
    diffusers/examples/unconditional_image_generation/train_unconditional.py \
    --dataset_name="cifar10" \
    --model_config_name_or_path="google/ddpm-cifar10-32" \
    --resolution=32 \
    --output_dir="runs/ddpm-cifar10-repro" \
    --train_batch_size=64 \
    --num_epochs=10 \
    --learning_rate=1e-4 \
    --lr_warmup_steps=500 \
    --save_images_epochs=5 \
    --save_model_epochs=5
```

## 주요 해결한 문제들
1. **diffusers 버전 불일치**: 개발 버전(0.35.0.dev0) 설치로 해결
2. **tensorboard 누락**: `pip install tensorboard`로 해결  
3. **CIFAR-10 컬럼명 오류**: `"image"` → `"img"`로 수정

## 훈련 결과 확인
```bash
# 훈련 진행 상황 확인
ps aux | grep train_unconditional | grep -v grep

# 결과 폴더 확인  
ls -la runs/ddpm-cifar10-repro/

# tensorboard 로그 확인
ls -la runs/ddpm-cifar10-repro/logs/
```

## 예상 소요 시간
- 환경 설정: 5-10분
- 훈련 (10 epochs): 10-20분
- 총 소요 시간: 15-30분 