import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

def main():
    # ==========================================
    # 1. 실제 데이터셋 로드 & 프롬프트 포맷팅
    # ==========================================
    print("1. Hugging Face 허브에서 실제 KoAlpaca 데이터셋을 로드합니다...")
    dataset = load_dataset("Beomi/KoAlpaca-v1.1", split="train")
    
    # 모델이 대화 맥락을 이해할 수 있도록 프롬프트 형태로 변환하는 함수
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = (
                f"### 명령어:\n{example['instruction'][i]}\n\n"
                f"### 응답:\n{example['output'][i]}<|endoftext|>"
            )
            output_texts.append(text)
        return output_texts

    # ==========================================
    # 2. 베이스 모델 및 토크나이저 설정
    # ==========================================
    # 실습을 위해 가벼운 오픈소스 모델(예: Qwen 0.5B 또는 Gemma 2B 등)을 타겟으로 잡습니다.
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    print(f"2. 베이스 모델 로드 중: {model_id}")
    
    # GPU 메모리 절약을 위한 4비트 양자화 설정 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 구조적 오버랩 방지 고정

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # ==========================================
    # 3. LoRA (Low-Rank Adaptation) 설정
    # ==========================================
    # 일부 파라미터만 어댑터 형태로 학습하여 메모리 소모를 극적으로 줄입니다.
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 모델 아키텍처 내 타겟 레이어
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # ==========================================
    # 4. 학습 하이퍼파라미터(TrainingArguments) 설정
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./sft_output_model",     # 모델이 저장될 디렉토리
        num_train_epochs=1,                 # 실습을 위해 1 에포크만 수행
        per_device_train_batch_size=2,      # GPU 메모리에 맞춰 조절 (VRAM 8GB 기준 2~4)
        gradient_accumulation_steps=4,      # 배치 사이즈 보완 효과
        optim="paged_adamw_8bit",
        logging_steps=10,                   # 10 스텝마다 로그 출력
        learning_rate=2e-4,                 # LoRA 추천 학습률
        weight_decay=0.001,
        fp16=False,
        bf16=True,                          # Ampere 아키텍처 이상 GPU(RTX 30/40 시리즈 등) 필수
        max_steps=100,                      # 빠른 실습 테스트를 위해 100스텝에서 강제 종료 설정
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )

    # ==========================================
    # 5. SFTTrainer를 이용한 학습 파이프라인 결합
    # ==========================================
    print("3. 파인튜닝 파이프라인 초기화 및 학습을 시작합니다.")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,                 # 문장 최대 토큰 길이 제한
        tokenizer=tokenizer,
        args=training_args
    )

    # 학습 실행
    trainer.train()

    # ==========================================
    # 6. 학습 완료된 어댑터 모델 저장
    # ==========================================
    print("4. 학습 완료된 LoRA 가중치를 로컬에 저장합니다.")
    trainer.model.save_pretrained("./final_lora_adapter")
    tokenizer.save_pretrained("./final_lora_adapter")
    print("학습 및 저장이 완료되었습니다!")

if __name__ == "__main__":
    main()