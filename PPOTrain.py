import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import os

# --- 1. 환경 생성 및 검증 ---
# PPO 학습을 위해 Vectorized 환경을 사용하는 것이 일반적입니다.
# num_envs=1 로 설정하면 단일 AGV 환경이 됩니다.
# PPO는 기본적으로 병렬 환경에서 학습되므로, num_envs를 늘려 Multi-Agent 환경으로 확장 가능합니다.
num_parallel_envs = 1 # 일단 1로 시작, Multi-Agent로 확장 시 이 값 증가

# 환경 인스턴스 생성 (render_mode='human'은 학습 중 시각화용)
# 학습 시에는 render_mode=None으로 설정하여 불필요한 오버헤드 제거
env_id = "WarehouseEnv-v0" # 나중에 gym.register로 등록할 이름
# Gymnasium에 사용자 정의 환경 등록
gym.envs.registration.register(
    id=env_id,
    entry_point="your_module_name:WarehouseEnv", # 실제 파일명과 클래스명으로 변경 필요
    kwargs={'map_size': (20, 20), 'num_racks': 5, 'num_agvs': 1},
    max_episode_steps=500,
)

# 환경 유효성 검사
# from stable_baselines3.common.env_checker import check_env
# try:
#     check_env(WarehouseEnv(render_mode=None))
#     print("Environment check passed!")
# except Exception as e:
#     print(f"Environment check failed: {e}")

num_parallel_envs = 1 # 단일 AGV 시뮬레이션

vec_env = make_vec_env(lambda: gym.make(env_id, render_mode=None), n_envs=num_parallel_envs)
vec_env = VecMonitor(vec_env, filename="logs/ppo_warehouse_monitor_layout_based")

# --- 2. 모델 학습을 위한 하이퍼파라미터 설정 ---
# 이전 답변의 하이퍼파라미터와 동일하게 유지 (요청하신 초기 설정값)
hyperparameters = {
    "batch_size": 1024,
    "n_steps": 1024, # time_horizon
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_epochs": 3,
    "ent_coef": 1e-2, # beta (entropy coefficient)
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "tensorboard_log": "./ppo_warehouse_tensorboard_logs_layout_based/"
}

# --- 3. PPO 모델 생성 ---
# Observation Space가 Dict이므로 MultiInputPolicy 사용
model = PPO("MultiInputPolicy", 
            vec_env,
            verbose=1,
            seed=42,
            **hyperparameters
           )

# --- 4. 콜백 설정 (선택 사항이지만 권장) ---
log_dir = "./tmp/logs_layout_based/"
os.makedirs(log_dir, exist_ok=True)

stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=10, 
    min_evals=20, 
    verbose=1
)

eval_env = gym.make(env_id, render_mode=None) # 평가용 환경도 생성
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./best_models_layout_based/",
    log_path="./logs_layout_based/",
    eval_freq=10000, 
    deterministic=True, 
    render=False,
    callback_after_eval=stop_callback, 
    verbose=1
)

# --- 5. 모델 학습 ---
total_timesteps = int(1e6) # 테스트를 위해 일단 낮게 설정. 최종적으로는 1e7~1e8
print(f"Starting PPO training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps, callback=eval_callback)
print("Training finished.")

# --- 6. 학습된 모델 저장 ---
model.save("ppo_warehouse_optimized_agv_layout_based")
print("Model saved as ppo_warehouse_optimized_agv_layout_based.zip")

# --- 7. 학습된 모델 로드 및 테스트 (선택 사항) ---
print("\n--- Testing the trained model ---")
loaded_model = PPO.load("ppo_warehouse_optimized_agv_layout_based")

test_env = gym.make(env_id, render_mode="human") # 테스트 환경은 렌더링 켜서 확인
obs, info = test_env.reset()
done = False
episode_reward = 0
num_test_steps = 0

while not done and num_test_steps < test_env.max_episode_steps * 2: # 충분히 긴 스텝 테스트
    action, _states = loaded_model.predict(obs, deterministic=True) 
    obs, reward, terminated, truncated, info = test_env.step(action)
    done = terminated or truncated
    episode_reward += reward
    num_test_steps += 1
    
    if "event" in info:
        print(f"Event: {info['event']}")
    if info.get("collision", False): # 충돌 발생 시
        print(f"Collision detected at step {num_test_steps}")

print(f"Test Episode finished with reward: {episode_reward} in {num_test_steps} steps.")
test_env.close()

# TensorBoard로 학습 과정 시각화
# 터미널에서 실행: tensorboard --logdir ./ppo_warehouse_tensorboard_logs_layout_based/