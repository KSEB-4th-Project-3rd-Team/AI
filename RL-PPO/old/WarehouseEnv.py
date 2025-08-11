import numpy as np
import random
import collections
import gymnasium as gym
from gymnasium import spaces

class WarehouseEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, map_size=(41, 50), num_racks_per_row=6, num_rack_rows=4, render_mode=None):
        super(WarehouseEnv, self).__init__()

        self.map_size = map_size # (height, width) - 예시: 41x50 그리드
        self.num_racks_per_row = num_racks_per_row # 한 줄에 있는 랙 섹션 수
        self.num_rack_rows = num_rack_rows # 랙 줄 수 (Pallet Racking, Pick Shelving)
        self.render_mode = render_mode

        self.warehouse_map = np.zeros(map_size, dtype=int) # 0: 빈 공간, 1: 벽/고정 장애물, 2: 랙, 3: 출고존, 4: 입고존, 5: 충전소, 6: 사무실 등
        
        self.rack_positions = [] # (r, c, rack_id)
        self.rack_states = {} # rack_id: {"status": "empty"|"filled"|"waiting_delivery", "item_type": "tire_set"}
        self.agv_positions = [] # (r, c)

        self.rack_states = {} # rack_id: {"status": "empty"|"filled"|"waiting_delivery", "item_type": "tire_set"}
        self.agv_carrying_rack = [None] * num_agvs # AGV가 운반 중인 랙 ID (단일 AGV: [None])
        self.order_queue = collections.deque() # (order_id, required_rack_id, destination_zone) 또는 여러 랙을 위한 확장된 구조

        self._place_static_elements() # 정적 요소 배치

        # --- Observation Space 정의 (업데이트) ---
        # 맵 크기가 커지므로 전체 맵을 Observation으로 주는 것은 비효율적일 수 있습니다.
        # AGV 주변 부분 맵, AGV 위치, 목표 위치, 주요 랙들의 상태를 조합합니다.

        # 1. AGV 위치: (x, y) - 정수
        # 2. 목표 위치: (x, y) - 정수 (현재 AGV가 이동해야 할 랙 또는 존의 위치)
        # 3. 랙 상태: 모든 랙의 상태 (채움/빈자리/출고대기)
        #    - 랙 ID별 상태를 배열로 표현 (num_racks * 1)
        # 4. AGV 주변 맵 정보: AGV를 중심으로 한 작은 그리드 (예: 7x7)
        #    - 장애물, 다른 AGV, 랙, 존 등 포함
        # 5. 현재 주문 큐 정보: 큐의 길이, 다음 주문 랙 ID 등 요약 정보

        # 랙의 총 개수 계산
        total_num_racks = (self.num_racks_per_row * 2 + 2) * self.num_rack_rows # Pallet Racking과 Pick Shelving 고려 (대략적인 계산)
        
        self.observation_space = spaces.Dict({
            "agv_position": spaces.Box(low=0, high=max(map_size)-1, shape=(2,), dtype=np.int32),
            "target_position": spaces.Box(low=0, high=max(map_size)-1, shape=(2,), dtype=np.int32), # AGV의 현재 목표 위치
            "rack_statuses": spaces.Box(low=0, high=2, shape=(total_num_racks,), dtype=np.int32), # 0:빈자리, 1:채움, 2:출고대기
            "agv_carrying_rack": spaces.Discrete(total_num_racks + 1), # 0: 안들고 있음, 1~N: 랙 ID
            "current_order_count": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32), # 현재 큐에 있는 주문 수 (최대 10개 가정)
            "warehouse_map_local": spaces.Box(low=0, high=6, shape=(7, 7), dtype=np.int32) # AGV 주변 7x7 그리드 (더 넓게)
        })

        self.action_space = spaces.Discrete(7) # 0: 정지, 1: 상, 2: 하, 3: 좌, 4: 우, 5: Pickup_Rack, 6: Drop_Rack

        self.current_step = 0
        self.max_episode_steps = 1000 # 에피소드 최대 스텝 증가 (맵 크기 고려)
        self.initial_order_count = 5 # 초기 주문 수
        
        self.last_agv_pos = None # 불필요한 대기/빈번한 방향전환 패널티용
        self.same_pos_count = 0
        self.path_history = collections.deque(maxlen=5) # 이동 경로 저장 (비효율 경로 판별용)

    def _place_static_elements(self):
        # 맵 초기화
        self.warehouse_map.fill(0) # 모든 셀을 빈 공간으로 초기화
        self.rack_positions = []
        self.rack_states = {}
        rack_id_counter = 0

        # --- 이미지 레이아웃 기반 구역 설정 (상대적 위치) ---
        # 맵 크기 (41, 50)에 맞춰 대략적인 좌표를 설정합니다.

        # 1. Warehouse Office (고정 장애물/벽)
        office_start_r, office_end_r = int(self.map_size[0] * 0.7), int(self.map_size[0] * 0.9)
        office_start_c, office_end_c = int(self.map_size[1] * 0.35), int(self.map_size[1] * 0.55)
        self.warehouse_map[office_start_r:office_end_r, office_start_c:office_end_c] = 1 # 벽/장애물

        # 2. Returns & Reprocessing (작업 공간 / 장애물로 간주)
        returns_start_r, returns_end_r = int(self.map_size[0] * 0.4), int(self.map_size[0] * 0.9)
        returns_start_c, returns_end_c = int(self.map_size[1] * 0.05), int(self.map_size[1] * 0.25)
        self.warehouse_map[returns_start_r:returns_end_r, returns_start_c:returns_end_c] = 1 # 장애물/작업 공간

        # 3. MHS Charging (충전소)
        charging_start_r, charging_end_r = int(self.map_size[0] * 0.7), int(self.map_size[0] * 0.9)
        charging_start_c, charging_end_c = int(self.map_size[1] * 0.9), int(self.map_size[1] * 0.98)
        self.warehouse_map[charging_start_r:charging_end_r, charging_start_c:charging_end_c] = 5 # 충전소 (특정 존)

        # 4. Inbound Zone (입고존)
        inbound_start_r, inbound_end_r = int(self.map_size[0] * 0.6), int(self.map_size[0] * 0.7)
        inbound_start_c, inbound_end_c = int(self.map_size[1] * 0.3), int(self.map_size[1] * 0.4)
        self.receiving_zone_pos = (inbound_start_r + (inbound_end_r - inbound_start_r)//2, inbound_start_c + (inbound_end_c - inbound_start_c)//2)
        self.warehouse_map[inbound_start_r:inbound_end_r, inbound_start_c:inbound_end_c] = 4 # 입고존
        
        # 5. Despatch Zone (출고존)
        despatch_start_r, despatch_end_r = int(self.map_size[0] * 0.6), int(self.map_size[0] * 0.7)
        despatch_start_c, despatch_end_c = int(self.map_size[1] * 0.6), int(self.map_size[1] * 0.7)
        self.delivery_zone_pos = (despatch_start_r + (despatch_end_r - despatch_start_r)//2, despatch_start_c + (despatch_end_c - despatch_start_c)//2)
        self.warehouse_map[despatch_start_r:despatch_end_r, despatch_start_c:despatch_end_c] = 3 # 출고존

        # 6. Pallet Racking & Pick Shelving (랙)
        # 이미지에 랙들이 길게 늘어서 있고, 사이에 통로가 있는 형태로 구성
        # 대략적인 랙 위치를 계산하여 배치합니다.
        rack_base_r = int(self.map_size[0] * 0.05) # 맵 상단에서 시작
        rack_base_c = int(self.map_size[1] * 0.15) # 맵 좌측에서 시작

        rack_width = 3 # 랙의 너비 (그리드 칸 수)
        rack_length = int(self.map_size[0] * 0.3) # 랙의 길이 (그리드 칸 수)
        aisle_width = 4 # 통로의 너비 (그리드 칸 수)

        for row_idx in range(self.num_rack_rows):
            current_r = rack_base_r
            # Pallet Racking (좌측 랙)
            for rack_col_idx in range(self.num_racks_per_row):
                start_c = rack_base_c + (rack_width + aisle_width) * rack_col_idx
                for r in range(current_r, current_r + rack_length):
                    for c in range(start_c, start_c + rack_width):
                        if 0 <= r < self.map_size[0] and 0 <= c < self.map_size[1]:
                            self.warehouse_map[r, c] = 2
                            self.rack_positions.append((r, c, rack_id_counter))
                            self.rack_states[rack_id_counter] = {"status": "filled", "item_type": "tire_set"}
                            rack_id_counter += 1
            
            # Pick Shelving (우측 랙) - Pallet Racking과 유사한 구조이지만 간격이 다를 수 있음
            # 일단 유사하게 배치하고, 필요시 간격 조정
            current_c_shelving = rack_base_c + (rack_width + aisle_width) * self.num_racks_per_row + aisle_width # Pallet Racking 끝 + 통로
            for rack_col_idx in range(2): # 이미지상 2개의 큰 섹션
                start_c = current_c_shelving + (rack_width + aisle_width) * rack_col_idx
                for r in range(current_r, current_r + rack_length):
                    for c in range(start_c, start_c + rack_width):
                        if 0 <= r < self.map_size[0] and 0 <= c < self.map_size[1]:
                            self.warehouse_map[r, c] = 2
                            self.rack_positions.append((r, c, rack_id_counter))
                            self.rack_states[rack_id_counter] = {"status": "filled", "item_type": "tire_set"}
                            rack_id_counter += 1
            
            # 다음 랙 줄의 시작 위치 조정 (줄 간 간격 고려)
            rack_base_r += rack_length + int(self.map_size[0] * 0.05) # 다음 랙 줄 시작 위치

        self.num_racks = rack_id_counter # 실제 배치된 랙의 총 개수로 업데이트

        # 맵 상의 통로 부분은 0 (빈 공간)으로 유지되어야 합니다.
        # 고정 장애물은 1로 설정합니다.

    def _get_obs(self):
        # 유니티에서 받아올 데이터를 바탕으로 Observation을 구성합니다.
        # 실제 환경에서는 AGV 위치, 센서 값, 랙 상태 등 모든 정보가 실시간으로 전송됩니다.
        
        agv_pos = np.array(self.agv_positions[0], dtype=np.int32)
        
        # 현재 AGV의 목표 위치 결정 (가장 우선순위 높은 작업의 목표)
        target_pos = np.array([-1,-1], dtype=np.int32) # 기본값: 목표 없음
        
        if self.agv_carrying_rack[0] is not None:
            # 랙을 운반 중이면 출고존이 목표
            target_pos = np.array(self.delivery_zone_pos, dtype=np.int32)
        elif len(self.order_queue) > 0:
            # 주문 큐에 랙 출고 주문이 있다면 해당 랙이 목표
            target_rack_id, _ = self.order_queue[0]
            for r_pos, r_id in [(rp[:2], rp[2]) for rp in self.rack_positions]:
                if r_id == target_rack_id:
                    target_pos = np.array(r_pos, dtype=np.int32)
                    break
        else:
            # 빈 랙이 있다면 입고존이 목표 (채우기 위해)
            empty_racks = [rid for rid, state in self.rack_states.items() if state["status"] == "empty"]
            if empty_racks:
                target_pos = np.array(self.receiving_zone_pos, dtype=np.int32)

        # 랙 상태 배열 (랙 ID 순서대로)
        rack_status_values = [
            (0 if self.rack_states[i]["status"] == "empty" else (1 if self.rack_states[i]["status"] == "filled" else 2))
            for i in range(self.num_racks)
        ]
        rack_statuses_array = np.array(rack_status_values, dtype=np.int32)

        # AGV가 들고 있는 랙 ID (들고 있지 않으면 0)
        agv_carrying_id = self.agv_carrying_rack[0] + 1 if self.agv_carrying_rack[0] is not None else 0
        
        # 현재 주문 큐 길이
        current_order_count = np.array([len(self.order_queue)], dtype=np.int32)

        # AGV 주변 맵 정보 (7x7 그리드)
        pad_size = 3 # 7x7 이므로 중앙에서 상하좌우 3칸
        padded_map = np.pad(self.warehouse_map, pad_size, mode='constant', constant_values=0)
        
        # 현재 AGV 위치가 (r, c)일 때, 패딩된 맵에서의 시작점 계산
        start_r_padded = agv_pos[0] + pad_size - pad_size
        start_c_padded = agv_pos[1] + pad_size - pad_size
        
        warehouse_map_local = padded_map[start_r_padded : start_r_padded + 7, 
                                         start_c_padded : start_c_padded + 7]

        return {
            "agv_position": agv_pos,
            "target_position": target_pos,
            "rack_statuses": rack_statuses_array,
            "agv_carrying_rack": np.array([agv_carrying_id], dtype=np.int32), # scalar를 array로
            "current_order_count": current_order_count,
            "warehouse_map_local": warehouse_map_local
        }

    # reset, step, _render_frame, close 메서드는 이전 답변과 거의 동일하게 유지
    # 단, 보상 함수에서 '불필요한 대기/빈번한 방향 전환' 로직 추가
def step(self, action):
    reward = -1.0 # 스텝당 소모 패널티
    terminated = False
    truncated = False
    info = {}

    agv_idx = 0 # 단일 AGV (AGV ID 0)
    current_pos = list(self.agv_positions[agv_idx])
    new_pos = list(current_pos) # 이동 액션 시 새로운 위치가 될 곳

    # 0~4: 이동 액션
    if action in [0, 1, 2, 3, 4]:
        if action == 0: new_pos[0] -= 1 # 상
        elif action == 1: new_pos[0] += 1 # 하
        elif action == 2: new_pos[1] -= 1 # 좌
        elif action == 3: new_pos[1] += 1 # 우
        # action == 4: 정지 (new_pos = current_pos)

        # 경계 체크 및 장애물/벽 충돌
        is_colliding = False
        if not (0 <= new_pos[0] < self.map_size[0] and 0 <= new_pos[1] < self.map_size[1]):
            is_colliding = True
        elif self.warehouse_map[new_pos[0], new_pos[1]] == 1: # 장애물/벽
            is_colliding = True
        
        if is_colliding:
            reward -= 100 # 충돌 패널티
            new_pos = current_pos # 이동 취소
            info["collision"] = True
        else:
            self.agv_positions[agv_idx] = new_pos
            info["collision"] = False

        # 불필요한 대기/빈번한 방향 전환 패널티 
        if tuple(new_pos) == tuple(current_pos): # 정지했거나 이동 못했을 때
            self.same_pos_count += 1
            if self.same_pos_count > 5: # 5스텝 이상 같은 위치에 있으면 불필요한 대기
                reward -= 0.5
        else:
            self.same_pos_count = 0

        self.path_history.append(tuple(new_pos))
        if len(self.path_history) == self.path_history.maxlen and len(set(self.path_history)) < 3: # 5스텝 중 3개 미만의 고유 위치 (빙글빙글 돌거나 왔다갔다)
            reward -= 0.5 # 빈번한 방향 전환/비효율 경로

    # 5: Pickup_Rack 액션
    elif action == 5:
        # AGV가 랙을 운반 중이 아닐 때만 픽업 시도
        if self.agv_carrying_rack[agv_idx] is None:
            # 현재 AGV 위치에 있는 랙을 찾음
            rack_at_agv_pos = None
            for r_pos, r_c, r_id in self.rack_positions: # r_pos, r_c는 랙의 (행, 열) 위치
                if (r_pos, r_c) == tuple(current_pos):
                    rack_at_agv_pos = r_id
                    break

            if rack_at_agv_pos is not None:
                rack_status = self.rack_states[rack_at_agv_pos]["status"]
                
                # 1) 출고 대기 중인 랙을 픽업
                if rack_status == "waiting_delivery":
                    self.agv_carrying_rack[agv_idx] = rack_at_agv_pos
                    # 랙의 상태는 그대로 "waiting_delivery" (픽업했으나 아직 출고 안됨)
                    info["event"] = f"picked_up_delivery_rack_{rack_at_agv_pos}"
                    # print(f"AGV picked up rack {rack_at_agv_pos} for delivery.")
                    # 픽업 성공 보상 (중간 과정이므로 낮은 보상 또는 없음)
                    reward += 10 # 픽업 액션 자체에 대한 보상
                
                # 2) 입고존에서 새로운 랙을 픽업하여 빈 랙 위치로 운반 (replenish task)
                elif tuple(current_pos) == self.receiving_zone_pos:
                    # 입고존에 실제 랙이 존재해야 함 (또는 가상으로 랙이 생성됨)
                    # 여기서는 '빈 랙을 채울 타이어 세트가 있는 랙'으로 가정
                    # 입고존에서 픽업 가능한 (새로운) 랙이 있다고 가정하고 픽업
                    self.agv_carrying_rack[agv_idx] = rack_at_agv_pos # 입고존의 랙 ID
                    self.rack_states[rack_at_agv_pos]["status"] = "in_transit" # 운반 중 상태
                    info["event"] = f"picked_up_replenish_rack_{rack_at_agv_pos}"
                    # print(f"AGV picked up rack {rack_at_agv_pos} from receiving zone for replenishment.")
                    reward += 10 # 픽업 성공 보상
                
                else:
                    # 잘못된 랙 픽업 시도 (픽업할 필요 없는 랙)
                    reward -= 1 # 작은 패널티 (불필요한 액션)
                    info["event"] = "invalid_pickup_attempt_rack_not_ready"
            else:
                # 랙이 없는 위치에서 픽업 시도
                reward -= 1 # 작은 패널티
                info["event"] = "invalid_pickup_attempt_no_rack"
        else:
            # 이미 랙을 운반 중인데 픽업 시도
            reward -= 1 # 작은 패널티
            info["event"] = "invalid_pickup_attempt_already_carrying"

    # 6: Drop_Rack 액션
    elif action == 6:
        # AGV가 랙을 운반 중일 때만 드롭 시도
        if self.agv_carrying_rack[agv_idx] is not None:
            carried_rack_id = self.agv_carrying_rack[agv_idx]
            
            # 1) 출고존에 랙 드롭
            if tuple(current_pos) == self.delivery_zone_pos:
                # 현재 주문 큐의 맨 앞 주문과 일치하는 랙인지 확인
                if len(self.order_queue) > 0 and self.order_queue[0]["required_rack_id"] == carried_rack_id:
                    self.order_queue.popleft() # 주문 완료
                    self.rack_states[carried_rack_id]["status"] = "empty" # 랙 빈자리
                    self.agv_carrying_rack[agv_idx] = None # 랙 내려놓음
                    reward += 100 # 성공적 주문 출고
                    info["event"] = f"order_completed_rack_{carried_rack_id}"
                    # print(f"Order for rack {carried_rack_id} completed. Reward: {reward}")
                    
                else:
                    # 엉뚱한 랙을 출고존에 내려놓음 (패널티)
                    reward -= 50
                    self.agv_carrying_rack[agv_idx] = None # 일단 내려놓게 함
                    info["event"] = f"wrong_dropoff_delivery_zone_{carried_rack_id}"
                    # print(f"Wrong rack {carried_rack_id} dropped at delivery zone. Penalty: {reward}")
            
            # 2) 빈 랙 위치에 랙 드롭 (재고 보충 완료)
            else: # 현재 위치가 출고존이 아닌 랙 위치일 경우 (빈 랙 채우기)
                # 현재 AGV 위치가 'empty' 랙의 위치이고, 이 랙이 채워져야 할 랙일 경우
                rack_at_agv_pos_id = None
                for r_pos, r_c, r_id in self.rack_positions:
                    if (r_pos, r_c) == tuple(current_pos) and self.rack_states[r_id]["status"] == "empty":
                        rack_at_agv_pos_id = r_id
                        break
                
                if rack_at_agv_pos_id is not None and carried_rack_id == rack_at_agv_pos_id:
                    # AGV가 입고존에서 픽업한 랙을 빈 랙 위치에 정확히 가져왔을 때
                    self.rack_states[carried_rack_id]["status"] = "filled" # 랙 채움 성공
                    self.agv_carrying_rack[agv_idx] = None # 랙 내려놓음
                    reward += 50 # 랙 채움 성공 보상
                    info["event"] = f"rack_filled_at_position_{carried_rack_id}"
                    # print(f"Rack {carried_rack_id} filled at its position. Reward: {reward}")
                    # 채움 완료 후 새로운 주문 생성 (실제로는 WMS에서 관리)
                    if not self.order_queue: # 큐가 비어있고, 모든 랙이 'filled'가 아닌 경우에만
                        self._generate_new_order()
                else:
                    # 엉뚱한 위치에 랙을 내려놓음 또는 픽업한 랙이 빈 랙과 일치하지 않음
                    reward -= 10 # 중간 패널티
                    info["event"] = f"invalid_dropoff_location_{carried_rack_id}"
                    # print(f"Invalid drop-off location for rack {carried_rack_id}. Penalty: {reward}")

        else:
            # 랙을 운반 중이 아닌데 드롭 시도
            reward -= 1 # 작은 패널티
            info["event"] = "invalid_dropoff_attempt_not_carrying"

    # 비효율 경로/장시간 미완수 (current_step 기반으로 패널티)
    self.current_step += 1
    if self.current_step % 200 == 0:
        if len(self.order_queue) > 0 and not info.get("order_completed_rack", False) and not info.get("rack_filled_at_position", False):
            reward -= 10 # 장시간 미완수 패널티

    # 모든 주문 완료 (입력된 리스트 완료)
    # 모든 출고 주문이 완료되고, 모든 랙이 'filled' 상태이거나, AGV가 운반 중인 랙이 없다면
    if not self.order_queue and all(state["status"] in ["filled", "empty"] for state in self.rack_states.values()):
        if self.agv_carrying_rack[agv_idx] is None:
            reward += 500
            terminated = True
            info["event"] = "all_orders_and_replenishments_completed"
            # print("All orders and replenishments completed! Episode terminated.")

    if self.current_step >= self.max_episode_steps:
        truncated = True
        info["event"] = "time_truncated"
        # print("Episode truncated due to max steps.")

    observation = self._get_obs()

    if self.render_mode == "human":
        self._render_frame()

    return observation, reward, terminated, truncated, info

# WarehouseEnv 클래스의 _get_obs 메서드 내부에
def _get_obs(self):
    agv_pos = np.array(self.agv_positions[0], dtype=np.int32)
    
    # AGV의 현재 목표 위치 결정 (가장 우선순위 높은 작업의 목표)
    target_pos = np.array([-1,-1], dtype=np.int32) # 기본값: 목표 없음
    
    if self.agv_carrying_rack[0] is not None:
        # 랙을 운반 중이면, 랙의 상태에 따라 목적지가 달라짐
        carried_rack_id = self.agv_carrying_rack[0]
        if self.rack_states[carried_rack_id]["status"] == "waiting_delivery":
            target_pos = np.array(self.delivery_zone_pos, dtype=np.int32) # 출고존으로
        elif self.rack_states[carried_rack_id]["status"] == "in_transit":
            # 입고존에서 픽업한 랙이라면, 빈 랙 위치가 목표
            empty_rack_pos = None
            for r_pos, r_c, r_id in self.rack_positions:
                if self.rack_states[r_id]["status"] == "empty": # 가장 먼저 발견된 빈 랙으로
                    empty_rack_pos = (r_pos, r_c)
                    break
            if empty_rack_pos:
                target_pos = np.array(empty_rack_pos, dtype=np.int32)
            else: # 빈 랙이 없으면 입고존으로 다시 돌아가거나 대기
                target_pos = np.array(self.receiving_zone_pos, dtype=np.int32)

    elif len(self.order_queue) > 0:
        # 주문 큐에 출고 주문이 있다면 해당 랙이 목표
        current_order = self.order_queue[0]
        target_rack_id = current_order["required_rack_id"]
        # 실제 랙 위치 목록에서 해당 랙 ID의 위치를 찾음
        for r_pos, r_c, r_id in self.rack_positions:
            if r_id == target_rack_id:
                target_pos = np.array((r_pos, r_c), dtype=np.int32)
                break
    else:
        # 주문 큐가 비어있고, 빈 랙이 있다면 입고존이 목표 (채우기 위해)
        empty_racks_exist = any(state["status"] == "empty" for state in self.rack_states.values())
        if empty_racks_exist:
            target_pos = np.array(self.receiving_zone_pos, dtype=np.int32)

    # 랙 상태 배열 (랙 ID 순서대로)
    # 0: empty, 1: filled, 2: waiting_delivery, 3: in_transit (새로운 상태)
    rack_status_values = []
    for i in range(self.num_racks):
        status = self.rack_states[i]["status"]
        if status == "empty": rack_status_values.append(0)
        elif status == "filled": rack_status_values.append(1)
        elif status == "waiting_delivery": rack_status_values.append(2)
        elif status == "in_transit": rack_status_values.append(3) # 새로운 상태 추가
    rack_statuses_array = np.array(rack_status_values, dtype=np.int32)

    # AGV가 들고 있는 랙 ID (들고 있지 않으면 0)
    # AGV_carrying_rack은 랙 ID가 0부터 시작하므로, 0을 들고 있지 않음을 의미
    agv_carrying_id = self.agv_carrying_rack[0] + 1 if self.agv_carrying_rack[0] is not None else 0
    
    # 현재 주문 큐 길이
    current_order_count = np.array([len(self.order_queue)], dtype=np.int32)

    # AGV 주변 맵 정보 (7x7 그리드)
    pad_size = 3
    padded_map = np.pad(self.warehouse_map, pad_size, mode='constant', constant_values=0)
    start_r_padded = agv_pos[0] + pad_size - pad_size
    start_c_padded = agv_pos[1] + pad_size - pad_size
    warehouse_map_local = padded_map[start_r_padded : start_r_padded + 7, 
                                     start_c_padded : start_c_padded + 7]

    return {
        "agv_position": agv_pos,
        "target_position": target_pos,
        "rack_statuses": rack_statuses_array,
        "agv_carrying_rack": np.array([agv_carrying_id], dtype=np.int32),
        "current_order_count": current_order_count,
        "warehouse_map_local": warehouse_map_local
    }

# WarehouseEnv 클래스의 _generate_new_order 메서드 내부에
def _generate_new_order(self):
    # 새로운 출고 주문 생성 (filled 상태의 랙 선택)
    filled_racks = [i for i, state in self.rack_states.items() if state["status"] == "filled"]
    if filled_racks:
        new_order_rack_id = random.choice(filled_racks)
        self.order_queue.append({
            "order_id": f"ORD_{self.current_step}_new",
            "required_rack_id": new_order_rack_id,
            "destination_zone": "despatch"
        })
        self.rack_states[new_order_rack_id]["status"] = "waiting_delivery"
        # print(f"New order generated for rack {new_order_rack_id}")
    #     cell_size = self.window_size // max(self.map_size)

    #     # Draw warehouse map (장애물, 랙, 존)
    #     for r in range(self.map_size[0]):
    #         for c in range(self.map_size[1]):
    #             rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
    #             color = (0,0,0) # Default for empty space
    #             if self.warehouse_map[r, c] == 1: # Obstacle/Wall/Office/Returns
    #                 color = (128, 128, 128) # Grey
    #             elif self.warehouse_map[r, c] == 2: # Rack
    #                 rack_id_at_pos = None
    #                 for rp_r, rp_c, rp_id in self.rack_positions:
    #                     if rp_r == r and rp_c == c:
    #                         rack_id_at_pos = rp_id
    #                         break
    #                 if rack_id_at_pos is not None:
    #                     if self.rack_states[rack_id_at_pos]["status"] == "filled":
    #                         color = (0, 0, 200) # Dark Blue (Filled)
    #                     elif self.rack_states[rack_id_at_pos]["status"] == "empty":
    #                         color = (255, 165, 0) # Orange (Empty)
    #                     else: # waiting_delivery
    #                         color = (255, 0, 255) # Magenta (Waiting)
    #             elif self.warehouse_map[r, c] == 3: # Despatch Zone (Delivery)
    #                 color = (0, 200, 0) # Dark Green
    #             elif self.warehouse_map[r, c] == 4: # Inbound Zone (Receiving)
    #                 color = (200, 200, 0) # Dark Yellow
    #             elif self.warehouse_map[r, c] == 5: # Charging Station
    #                 color = (0, 255, 255) # Cyan
                
    #             pygame.draw.rect(canvas, color, rect)
    #             pygame.draw.rect(canvas, (0, 0, 0), rect, 1) # Black border

    #     # Draw AGV
    #     for agv_pos in self.agv_positions:
    #         agv_rect = pygame.Rect(agv_pos[1] * cell_size, agv_pos[0] * cell_size, cell_size, cell_size)
    #         pygame.draw.rect(canvas, (255, 0, 0), agv_rect) # Red AGV

    #     self.window.blit(canvas, canvas.get_rect())
    #     pygame.event.pump()
    #     pygame.display.update()
    #     self.clock.tick(self.metadata["render_fps"])

    # def close(self):
    #     if self.window is not None:
    #         import pygame
    #         pygame.display.quit()
    #         pygame.quit()