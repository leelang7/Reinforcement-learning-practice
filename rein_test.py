import os
# OpenMP 중복 로드 충돌 방지 설정 추가
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


# 1. 가상의 토크나이저 로드 (실습을 위해 널리 쓰이는 Llama/Gemma 등의 구조 가정)
# 여기서는 예시로 로컬 환경에서 가볍게 쓸 수 있는 BERT 또는 기본 토크나이저를 사용합니다.
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# ==========================================
# [실습 1] 모방학습(SFT) 데이터 가공 및 토큰화
# ==========================================
'''
sft_raw_data = [
    {"instruction": "점심 메뉴 추천해줘. 매운 건 못 먹어.", "output": "매운 음식을 제외한다면 담백한 돈가스를 추천합니다."}
]
'''
sft_raw_data = load_dataset("Beomi/KoAlpaca-v1.1a", split="train")

def preprocess_sft(examples):
    # 질문과 정답을 하나의 대화 포맷 텍스트로 결합
    texts = [f"질문: {inst}\n답변: {out}" for inst, out in zip(examples["instruction"], examples["output"])]
    # 토큰화 수행 (최대 길이 제한 및 패딩 추가)
    model_inputs = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    
    # 지도학습에서는 입력 토큰(input_ids) 자체가 곧 예측 대상(labels)이 됩니다.
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

sft_dataset = Dataset.from_list(sft_raw_data)
sft_tokenized = sft_dataset.map(preprocess_sft, batched=True)

print("--- 모방학습(SFT) 데이터셋 1번 샘플 텐서 변환 결과 ---")
print("Input IDs:", sft_tokenized[0]["input_ids"][:10], "... (이하 패딩)")


# ==========================================
# [실습 2] 강화학습(DPO/RL) 데이터 가공 및 토큰화
# ==========================================
pref_raw_data = [
    {
        "prompt": "점심 메뉴 추천해줘. 매운 건 못 먹어.",
        "chosen": "매운 맛을 피하신다면 바삭한 돈가스를 추천합니다.",
        "rejected": "스트레스 풀리는 매운 떡볶이 어떠신가요?"
    }
]

def preprocess_preference(examples):
    # 강화학습(DPO 등) 모델은 프롬프트, 좋은 답변, 나쁜 답변을 각각 분리하여 토큰화한 뒤
    # 좋은 답변의 확률은 높이고, 나쁜 답변의 확률은 낮추는 loss를 계산합니다.
    tokenized_prompt = tokenizer(examples["prompt"], padding="max_length", max_length=32, truncation=True)
    tokenized_chosen = tokenizer(examples["chosen"], padding="max_length", max_length=32, truncation=True)
    tokenized_rejected = tokenizer(examples["rejected"], padding="max_length", max_length=32, truncation=True)
    
    return {
        "prompt_ids": tokenized_prompt["input_ids"],
        "chosen_ids": tokenized_chosen["input_ids"],
        "rejected_ids": tokenized_rejected["input_ids"]
    }

pref_dataset = Dataset.from_list(pref_raw_data)
pref_tokenized = pref_dataset.map(preprocess_preference, batched=True)

print("\n--- 강화학습(Preference) 데이터셋 1번 샘플 텐서 변환 결과 ---")
print("Prompt IDs:  ", pref_tokenized[0]["prompt_ids"][:5])
print("Chosen IDs:  ", pref_tokenized[0]["chosen_ids"][:5])
print("Rejected IDs:", pref_tokenized[0]["rejected_ids"][:5])

# 첫 번째 데이터 확인하기
print("--- 실제 SFT 데이터 샘플 ---")
print("질문 (Instruction):", sft_dataset[0]["instruction"])
print("답변 (Output):", sft_dataset[0]["output"])