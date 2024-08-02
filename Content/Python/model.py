import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import unreal

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #  torch.cuda.is_available() GPU를 사용가능하면 True, 아니라면 False를 리턴

# 강화학습 모델 설계
# 상태(state)는 엔드이펙터의 위치와 돌아간 정도 (Location, Rotation)
# 액션은 케이블 길이 8, 각 케이블 장력 8

# ---------------------------------------------------------------------------------------------
class PolicyNet(nn.Module): # 행위자 네트워크 정의
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(6, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.softmax(self.l5(x), dim=1)
        return x


class ValueNet(nn.Module): # 비평자 네트워크 정의
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(6, 128)
        self.l1 = nn.Linear(64, 128)
        self.l1 = nn.Linear(128, 128)
        self.l1 = nn.Linear(128, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.9  # 할인율 정의
        self.lr_pi = 0.0002  # 행위자 학습률
        self.lr_v = 0.0005  # 비평자 학습률
        self.action_size = 16  # 액션은 반드시 케이블 길이와 텐션값

        self.pi = PolicyNet(self.action_size).to(device)  
        self.v = ValueNet().to(device)  

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)  # 행위자 옵티마이저
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)  # 비평자 옵티마이저

    def get_action(self, state):
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)  # 상태를 텐서로 변환
        probs = self.pi(state)  # 행위자 네트워크로 확률 분포 계산
        probs = probs[0]  # 배치 차원 제거
        m = Categorical(probs)  # 범주형 분포 생성
        action = m.sample().item()  # 행동 샘플링
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        # 여기서 받는 action_prob는 행위자가 행동을 한 뒤에 나오는 상태이다. 즉, 이미 행동을 해서 환경과 상호작용을 한 것 따라서 환경만으로 비평자가 평가할 수 있다
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32).to(device)  # 현재 상태 텐서
        next_state = torch.tensor(next_state[np.newaxis, :], dtype=torch.float32).to(device)  # 다음 상태 텐서

        target = reward + self.gamma * self.v(next_state) * (1 - done)  # 목표값 계산
        target = target.detach()  # 목표값을 상수로 취급
        v = self.v(state)  # 현재 상태 가치 계산
        loss_fn = nn.MSELoss()  # 손실 함수 정의
        loss_v = loss_fn(v, target)  # 비평자 손실 계산 비평자는 기본적으로 행위자의 행동을 잘 평가해야한다. 

        delta = target - v  # TD 오차 계산
        loss_pi = -torch.log(action_prob) * delta.item()  # 행위자 손실 계산 (경사 상승법 위해서 -를 더 붙인다.)

        self.optimizer_v.zero_grad()  # 비평자 옵티마이저 초기화
        self.optimizer_pi.zero_grad()  # 행위자 옵티마이저 초기화
        loss_v.backward()  # 비평자 손실 역전파
        loss_pi.backward()  # 행위자 손실 역전파
        self.optimizer_v.step()  # 비평자 옵티마이저 업데이트
        self.optimizer_pi.step()  # 행위자 옵티마이저 업데이트

# ---------------------------------------------------------------------------------------------
# 현재 월드 가져오기
current_world = unreal.EditorLevelLibrary.get_editor_world()

# 게임 모드 베이스 가져오기
game_mode_env = unreal.GameplayStatics.get_game_mode(current_world)
agent = Agent()  # 에이전트 초기화
reward_history = []  # 보상 기록

for episode in range(2000):  # 2000 에피소드 동안 학습
    state = game_mode_env.ResetEnvironment()  # 환경 초기화
    total_reward = 0.0  # 총 보상 초기화

    state = state[0] if isinstance(state, tuple) else state

    while not game_mode_env.done:
        action, prob = agent.get_action(state)  # 행동 및 확률 계산
        next_state, reward, done, info, _ = game_mode_env.step(action)  # 환경에 행동 적용

        agent.update(state, prob, reward, next_state, done)  # 네트워크 업데이트

        state = next_state  # 다음 상태로 전환
        total_reward += reward  # 총 보상 누적

    reward_history.append(total_reward)  # 보상 기록
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))  # 100 에피소드마다 출력