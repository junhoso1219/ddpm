#!/usr/bin/env bash
# ============================================================
# DDPM 베이스라인 훈련 완전 자동화 스크립트
# venv 활성화 이후 실행하는 모든 과정
# ============================================================

set -e  # 오류 발생시 즉시 중단

echo "🚀 DDPM 베이스라인 훈련 환경 설정 시작..."

# 1. 작업 디렉터리 이동
cd ~/projects/baselines-reproduction
source .venv/bin/activate

echo "✅ 가상환경 활성화 완료"

# 2. diffusers 라이브러리 클론
echo "📦 diffusers 라이브러리 클론 중..."
if [ ! -d "diffusers" ]; then
    git clone https://github.com/huggingface/diffusers.git
else
    echo "   - diffusers 이미 존재함"
fi

# 3. 필수 패키지 설치
echo "📚 필수 패키지 설치 중..."
pip install accelerate transformers datasets torchvision tensorboard

# 4. diffusers 개발 버전 설치 (중요!)
echo "🔧 diffusers 개발 버전 설치 중..."
pip uninstall diffusers -y
pip install git+https://github.com/huggingface/diffusers.git

# 5. accelerate 설정 파일 생성
echo "⚙️  accelerate 설정 생성 중..."
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << EOF
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

# 6. CIFAR-10 컬럼명 문제 수정
echo "🔨 CIFAR-10 컬럼명 이슈 수정 중..."
sed -i 's/examples\["image"\]/examples["img"]/g' diffusers/examples/unconditional_image_generation/train_unconditional.py

# 7. 훈련 결과 폴더 생성
echo "📁 훈련 결과 폴더 생성 중..."
mkdir -p runs/ddpm-cifar10-repro

echo "🎯 모든 설정 완료! DDPM 훈련을 시작합니다..."

# 8. DDPM 훈련 실행
echo "🚂 DDPM 훈련 시작..."
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

echo "✅ DDPM 훈련 완료!"
echo "📊 결과는 runs/ddpm-cifar10-repro/ 폴더에 저장되었습니다." 