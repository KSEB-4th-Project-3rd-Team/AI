import os

class Config:
    def __init__(self):
        # 경로 설정
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.MAP_PATH = os.path.join(self.DATA_DIR, "map.csv")
        self.EXP_NO = "NO.1"
        self.FILE_TYPES = {
            "fig": ".png",
            "anim": ".gif",
            "stat": ".csv",
            "log": ".txt",
        }
        
        # 에이전트/실험 파라미터
        self.N_AGENTS = 1
        self.AGENT_NAMES = [f"AGV_{i+1}" for i in range(self.N_AGENTS)]
        self.HEURISTIC_TYPE = "manhattan"   # "dijkstra"
        self.ALLOW_WAIT_ACTION = True
        self.NUM_LOOPS = 1    # 반복 실험 횟수
        self.MAX_EPISODES = 100
        self.MAX_TIMESTEP = 200
        self.RANDOM_SEED = 42

        # 시각화/저장 옵션
        self.SAVE_FIG = True            # 경로 시각화 이미지 저장 여부
        self.SAVE_ANIMATION = True      # 경로 애니메이션 저장 여부
        self.ANIMATION_INTERVAL = 300   # 애니메이션 프레임 간격(ms)
        self.SHOW_COLLISIONS = True     # 충돌 표시 여부
        self.FIG_DPI = 150              # 그림 저장시 해상도

        # 강화학습 파라미터
        # self.RL_ALGO = "ppo"
        # self.RL_LEARNING_RATE = 1e-4
        # self.RL_GAMMA = 0.99
        # self.RL_BATCH_SIZE = 64

        # 배치 실험
        # self.EXPERIMENTS = [
        #     {
        #         "name": "exp01_cbs_manhattan",
        #         "n_agents": 8,
        #         "heuristic": "manhattan",
        #         "map_file": self.MAP_PATH,
        #         # "task_file": self.TASK_PATH
        #     },
        # ]

    def get_exp_folder(self, loop=None):
        folder = f"{self.EXP_NO}_epi{self.MAX_EPISODES}_ts{self.MAX_TIMESTEP}_loops{self.NUM_LOOPS}"
        if loop is not None:
            folder += f"_loop{loop}"
        path = os.path.join(self.RESULTS_DIR, folder)
        os.makedirs(path, exist_ok=True)
        return path

    def get_result_path(self, kind, loop=None):
        ext = self.FILE_TYPES[kind]
        exp_dir = self.get_exp_folder(loop)
        fname = f"{kind}_epi{self.MAX_EPISODES}_ts{self.MAX_TIMESTEP}_loops{self.NUM_LOOPS}"
        if loop is not None:
            fname += f"_loop{loop}"
        fname += ext
        return os.path.join(exp_dir, fname)

config = Config()

# fig_path = config.get_result_path("fig", loop=loop)       # .../fig.png
# anim_path = config.get_result_path("anim", loop=loop)     # .../anim.gif
# stat_path = config.get_result_path("stat", loop=loop)     # .../stat.csv
# log_path = config.get_result_path("log", loop=loop)       # .../log.txt
# summary_path = config.get_result_path("summary", loop=loop)  # .../summary.csv
