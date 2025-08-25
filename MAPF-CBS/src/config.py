class Config:
    def __init__(self):
        # === 경로 ===
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, "results")
        self.MAP_PATH = os.path.join(self.DATA_DIR, "map.csv")
        self.ASSIGN_TASK_PATH = os.path.join(self.DATA_DIR, "assign_task2.csv")
        self.EXP_NO = "NO.4"
        self.FILE_TYPES = {"fig": ".png", "anim": ".gif", "stat": ".csv", "log": ".txt"}

        # === 에이전트/실험 파라미터 ===
        self.N_AGENTS = 8                    # 동시 8개 배치
        self.AGENT_NAMES = [f"AGV_{i+1}" for i in range(self.N_AGENTS)]
        self.HEURISTIC_TYPE = "dijkstra"     # 휴리스틱은 정확 기준 권장
        '''
        EFFICIENCY_BASELINE="manhattan"으로 두면 장애물 우회가 반영 안 돼서 효율(%)이 과대평가되었음 -> dijkstra
        '''                                                       
        self.ALLOW_WAIT_ACTION = True      
        self.NUM_LOOPS = 1                  
        self.MAX_EPISODES = 100
        self.MAX_TIMESTEP = 200
        self.RANDOM_SEED = 42

        self.AGENT_ASSIGN_MODE = "auto"
        # 아래 두 줄은 자동 결정이 필요 없으면 None로 유지
        self.N_AGENTS = self.N_AGENTS
        self.AGENT_NAMES = self.AGENT_NAMES

        # === splitting 정책 ===
        self.CONFLICT_RESOLUTION_POLICY = "standard"

        # === 실험 스케일/샘플링 ===
        self.MAX_TASKS_PER_AGENT = 200       # KPI만 뽑을 상한
        self.SAMPLE_MODE = "head"            # "head" | "random" | "all"
        self.SAMPLE_SEED = 42

        # === 산출물 정책 (지표 전용) ===
        self.SAVE_FIG = True            
        self.SAVE_ANIMATION = True          
        self.ANIMATE_FIRST_K = 1
        self.SAVE_EVERY_N = 50

        # === 시각화 옵션 (미사용) ===
        self.AGENT_MARKER_SIZE = 6
        self.AGENT_LINE_WIDTH = 1.2
        self.RACK_ALPHA = 0.25

        # === 최단경로 기준(경로 효율 계산) ===
        self.EFFICIENCY_BASELINE = "astar"   # ★ A*로 실제 최단거리

    # ----(유틸 생략 가능. 기존 그대로 사용)----
    def get_exp_folder(self, agent_num, max_tasks=None):
        parts = [self.EXP_NO, f"agent{self.N_AGENTS}"]
        if max_tasks is not None: parts.append(f"maxT{max_tasks}")
        parts.append(time.strftime("%Y%m%d"))
        folder = "_".join(parts)
        path = os.path.join(self.RESULTS_DIR, folder)
        os.makedirs(path, exist_ok=True)
        return path

    def get_result_path(self, kind, agent_num=None, max_tasks=None, extra=None):
        ext = self.FILE_TYPES[kind]
        exp_dir = self.get_exp_folder(agent_num, max_tasks)
        fname_parts = [kind, f"agent{self.N_AGENTS}"]
        if max_tasks is not None: fname_parts.append(f"maxT{max_tasks}")
        if extra: fname_parts.append(str(extra))
        return os.path.join(exp_dir, "_".join(fname_parts) + ext)

config = Config()
