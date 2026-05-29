<<<<<<< HEAD
import os
# OpenMP 중복 로드 충돌 방지 설정 추가
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ========================================================
# 1. 가상의 언어 모델 클래스 정의 (이 부분이 반드시 위에 있어야 합니다)
# ========================================================
class SimpleLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)  # Vocabulary Size: 10
        
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

# ========================================================
# 2. 가상의 환경 및 보상 함수 (Reward Function) 정의
# ========================================================
# 규칙: 생성된 문장에 토큰 '5'가 포함되어 있으면 높은 보상(+10)을 지급
def get_reward(generated_sequence):
    if 5 in generated_sequence:
        return 10.0  # 좋은 답변
    return -2.0      # 잘못되거나 유해한 답변

# ========================================================
# 3. 강화학습(RL) 실행 메인 로직
# ========================================================
# 클래스가 위에 정의되었으므로 이제 NameError가 발생하지 않습니다.
rl_model = SimpleLanguageModel()
optimizer_rl = optim.Adam(rl_model.parameters(), lr=0.01)

prompt = torch.tensor([[1, 2, 3]])

print("--- 강화학습(RL) 시뮬레이션 시작 ---")
for episode in range(5):
    optimizer_rl.zero_grad()
    
    # 1. 모델이 프롬프트를 보고 스스로 답변 토큰들을 생성 (Exploration)
    logits = rl_model(prompt)
    probs = torch.softmax(logits, dim=-1) 
    
    generated_tokens = []
    log_probs = []
    
    # 문장 생성 (3개 토큰 샘플링)
    for t in range(3):
        dist = Categorical(probs[0, t])
        action = dist.sample()  # 확률 분포 기반 샘플링
        
        generated_tokens.append(action.item())
        log_probs.append(dist.log_prob(action))
        
    # 2. 생성된 문장에 대해 보상 평가
    reward = get_reward(generated_tokens)
    
    # 3. Policy Gradient Loss 계산 및 역전파
    total_log_prob = torch.stack(log_probs).sum()
    rl_loss = -total_log_prob * reward
    
    rl_loss.backward()
    optimizer_rl.step()
    
=======
import os
# OpenMP 중복 로드 충돌 방지 설정 추가
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ========================================================
# 1. 가상의 언어 모델 클래스 정의 (이 부분이 반드시 위에 있어야 합니다)
# ========================================================
class SimpleLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)  # Vocabulary Size: 10
        
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

# ========================================================
# 2. 가상의 환경 및 보상 함수 (Reward Function) 정의
# ========================================================
# 규칙: 생성된 문장에 토큰 '5'가 포함되어 있으면 높은 보상(+10)을 지급
def get_reward(generated_sequence):
    if 5 in generated_sequence:
        return 10.0  # 좋은 답변
    return -2.0      # 잘못되거나 유해한 답변

# ========================================================
# 3. 강화학습(RL) 실행 메인 로직
# ========================================================
# 클래스가 위에 정의되었으므로 이제 NameError가 발생하지 않습니다.
rl_model = SimpleLanguageModel()
optimizer_rl = optim.Adam(rl_model.parameters(), lr=0.01)

prompt = torch.tensor([[1, 2, 3]])

print("--- 강화학습(RL) 시뮬레이션 시작 ---")
for episode in range(5):
    optimizer_rl.zero_grad()
    
    # 1. 모델이 프롬프트를 보고 스스로 답변 토큰들을 생성 (Exploration)
    logits = rl_model(prompt)
    probs = torch.softmax(logits, dim=-1) 
    
    generated_tokens = []
    log_probs = []
    
    # 문장 생성 (3개 토큰 샘플링)
    for t in range(3):
        dist = Categorical(probs[0, t])
        action = dist.sample()  # 확률 분포 기반 샘플링
        
        generated_tokens.append(action.item())
        log_probs.append(dist.log_prob(action))
        
    # 2. 생성된 문장에 대해 보상 평가
    reward = get_reward(generated_tokens)
    
    # 3. Policy Gradient Loss 계산 및 역전파
    total_log_prob = torch.stack(log_probs).sum()
    rl_loss = -total_log_prob * reward
    
    rl_loss.backward()
    optimizer_rl.step()
    
>>>>>>> 693ea1fbfd07c85d2dbca7b81626097946835508
    print(f"Episode {episode+1} - 생성 문장 토큰: {generated_tokens}, 받은 보상: {reward:.1f}, Loss: {rl_loss.item():.4f}")