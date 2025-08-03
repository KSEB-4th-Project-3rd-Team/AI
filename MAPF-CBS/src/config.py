# src/config.py
import os

class Config:
    def __init__(self):
        # 경로 설정
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")

        self.MAP_PATH = os.path.join(self.DATA_DIR, "map.csv")
        
        # 에이전트/실험 파라미터
        self.N_AGENTS = 1
        self.AGENT_NAMES = [f"AGV_{i+1}" for i in range(self.N_AGENTS)]
        self.HEURISTIC_TYPE = "manhattan"   # "dijkstra"
        self.ALLOW_WAIT_ACTION = True
        self.MAX_EPISODES = 100
        self.MAX_TIMESTEP = 200
        self.RANDOM_SEED = 42
        self.EXPERIMENT_NAME = "exp01_baseline"

        # 시각화/저장 옵션
        self.SAVE_FIG = True
        self.SAVE_ANIMATION = True
        self.ANIMATION_INTERVAL = 300
        self.SHOW_COLLISIONS = True

        # 강화학습 파라미터
        # self.RL_ALGO = "ppo"
        # self.RL_LEARNING_RATE = 1e-4
        # self.RL_GAMMA = 0.99
        # self.RL_BATCH_SIZE = 64

        # 배치 실험 관리(여러 세트 돌릴 경우)
        self.EXPERIMENTS = [
            {
                "name": "exp01_cbs_manhattan",
                "n_agents": 8,
                "heuristic": "manhattan",
                "map_file": self.MAP_PATH,
                "task_file": self.TASK_PATH
            },
        
        ]

    def get_result_path(self, filename):
        return os.path.join(self.RESULTS_DIR, filename)

    def __repr__(self):
        return str(self.__dict__)

config = Config()
