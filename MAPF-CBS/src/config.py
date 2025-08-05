import os
import time
import pandas as pd

class Config:
    def __init__(self):
        # 경로 설정
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.MAP_PATH = os.path.join(self.DATA_DIR, "map.csv")
        self.ASSIGN_TASK_PATH = os.path.join(self.DATA_DIR, "assign_task.csv")
        self.EXP_NO = "NO.3"
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

        self.AGENT_ASSIGN_MODE = "auto"   # "auto"면 assign_task.csv로 자동 결정/ ( "single" / "multi")
        self.N_AGENTS = None              # assign_task.csv에서 자동 결정 (None이면)
        self.AGENT_NAMES = None           # 자동 결정 (None이면)

        # splitting 정책 (policy=standard)
        self.CONFLICT_RESOLUTION_POLICY = "standard" 

        # 시각화/저장 옵션
        self.SAVE_FIG = True            # 경로 시각화 이미지 저장 여부
        self.SAVE_ANIMATION = True      # 경로 애니메이션 저장 여부
        self.ANIMATION_INTERVAL = 20   # 애니메이션 프레임 간격(ms)
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

    def get_tasks_for_agent(self, agent_id, max_tasks=5, status_filter='WAIT'):
        '''
        agent_id: 에이전트 번호 (ex. 1)
        max_tasks: 최대 미션 개수 (ex. 5)
        status_filter: (기본 'WAIT' 미션만)
        '''
        df = pd.read_csv(self.ASSIGN_TASK_PATH)
        # assigned_agent가 agent_id이고 status가 'WAIT'인 것만
        tasks = df[(df['assigned_agent'] == agent_id) & (df['status'] == status_filter)]
        # 오더 순/우선순위 순서대로 최대 max_tasks개만
        tasks = tasks.sort_values(by=["order_item_id", "priority"]).head(max_tasks)
        return tasks

    def get_agents_and_tasks(self):
        mission_df = pd.read_csv(self.ASSIGN_TASK_PATH)
        agent_ids = sorted(mission_df['assigned_agent'].dropna().unique())
        n_agents = len(agent_ids)
        agent_names = [f"AGV_{i+1}" for i in range(n_agents)]
        starts, goals = [], []
        for agent_id in agent_ids:
            first_task = mission_df[mission_df['assigned_agent'] == agent_id].iloc[0]
            starts.append((int(first_task['start_y']), int(first_task['start_x'])))
            goals.append((int(first_task['goal_y']), int(first_task['goal_x'])))
        # 여기서 self.N_AGENTS 업데이트
        self.N_AGENTS = n_agents
        self.AGENT_NAMES = agent_names
        return n_agents, agent_names, starts, goals

    def get_exp_folder(self, agent_num, max_tasks=None):
        """
        저장 폴더명: 예) NO.3_agent2_maxT10_20240806/
        """
        parts = [self.EXP_NO]
        parts.append(f"agent{self.N_AGENTS}")
        if max_tasks is not None:
            parts.append(f"maxT{max_tasks}")
        parts.append(time.strftime("%Y%m%d"))
        folder = "_".join(parts)
        path = os.path.join(self.RESULTS_DIR, folder)
        os.makedirs(path, exist_ok=True)
        return path

    def get_result_path(self, kind, agent_num=None, max_tasks=None, extra=None):
        """
        저장 파일명: stat_agent2_maxT10.csv
        """
        ext = self.FILE_TYPES[kind]
        exp_dir = self.get_exp_folder(agent_num, max_tasks)
        fname_parts = [kind, f"agent{self.N_AGENTS}"]
        if max_tasks is not None:
            fname_parts.append(f"maxT{max_tasks}")
        if extra:
            fname_parts.append(str(extra))
        fname = "_".join(fname_parts) + ext
        return os.path.join(exp_dir, fname)

config = Config()

# fig_path = config.get_result_path("fig", loop=loop)       # .../fig.png
# anim_path = config.get_result_path("anim", loop=loop)     # .../anim.gif
# stat_path = config.get_result_path("stat", loop=loop)     # .../stat.csv
# log_path = config.get_result_path("log", loop=loop)       # .../log.txt
# summary_path = config.get_result_path("summary", loop=loop)  # .../summary.csv
