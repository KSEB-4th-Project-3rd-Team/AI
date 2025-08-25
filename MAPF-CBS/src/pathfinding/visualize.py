import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def split_full_path(full_path, splits):
    """
    full_path: [(y,x), ...]
    splits: [0, idx1, idx2, ..., N]
    → 태스크별 path: [full_path[splits[i]:splits[i+1]] for i in range(len(splits)-1)]
    """
    paths = []
    for i in range(len(splits)-1):
        seg = full_path[splits[i]:splits[i+1]]
        if seg:  # 빈 구간 예외 처리
            paths.append(seg)
    return paths

def plot_map_with_paths(
        my_map, full_path, task_splits,
        obstacles=None, racks=None, picking_zones=None, collisions=None,
        title='MAPF Paths', save_path=None):
    """
    full_path: 전체 경로 (list of (y, x))
    task_splits: 태스크 시작 인덱스 리스트 (예: [0, 100, 170, ...])  ※ 반드시 0부터 시작, 끝점 포함
    """
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap("Greys")
    plt.imshow(my_map, cmap=cmap, origin='lower')

    if racks:
        ys, xs = zip(*racks)
        plt.scatter(xs, ys, c='blue', marker='s', s=140, label='Rack')
    if picking_zones:
        ys, xs = zip(*picking_zones)
        plt.scatter(xs, ys, c='purple', marker='P', s=120, label='Picking Zone')

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 태스크별 path 분할
    paths = split_full_path(full_path, task_splits)
    for idx, path in enumerate(paths):
        if len(path) == 0:
            continue
        ys, xs = zip(*path)
        label = f'Task {idx+1}'
        color = color_cycle[idx % 5]  # 5색 순환
        plt.plot(xs, ys, marker='o', label=label, color=color, linewidth=2, markersize=4)
        plt.scatter([xs[0]], [ys[0]], color=color, s=50, marker='s')
        plt.scatter([xs[-1]], [ys[-1]], color=color, s=50, marker='*')

    # 태스크 경계점 표시(선택)
    for idx in range(1, len(task_splits)-1):
        y, x = full_path[task_splits[idx]]
        plt.scatter([x], [y], color='orange', s=60, marker='^', label='Task Switch' if idx==1 else "")

    if collisions:
        for col in collisions:
            if len(col['loc']) == 1:
                y, x = col['loc'][0]
                plt.scatter([x], [y], color='orange', s=160, marker='X', label='Collision')
            else:
                (y1, x1), (y2, x2) = col['loc']
                plt.plot([x1, x2], [y1, y2], color='orange', linewidth=3, linestyle='--')

    plt.title(title)
    plt.legend(loc='upper right', fontsize=9)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_single_path(my_map, path, agent_idx=0, obstacles=None, racks=None, picking_zones=None, title=None):
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap("Greys")
    plt.imshow(my_map, cmap=cmap, origin='lower')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ys, xs = zip(*path)
    plt.plot(xs, ys, marker='o', color=color_cycle[agent_idx % 5], linewidth=2)
    plt.scatter([xs[0]], [ys[0]], color=color_cycle[agent_idx % 5], s=120, marker='s')
    plt.scatter([xs[-1]], [ys[-1]], color=color_cycle[agent_idx % 5], s=120, marker='*')
    plt.title(title or f"Agent {agent_idx} Path")
    plt.gca().invert_yaxis()
    plt.show()

def animate_paths(
        my_map, full_path, task_splits, 
        obstacles=None, racks=None, picking_zones=None,
        interval=500, title='MAPF Animation', save_path=None):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    paths = split_full_path(full_path, task_splits)
    max_t = sum(len(p) for p in paths)
    seg_lengths = [len(p) for p in paths]
    seg_cumsum = np.cumsum([0]+seg_lengths)
    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.get_cmap("Greys")
    ax.imshow(my_map, cmap=cmap, origin='lower')
    lines = []
    for i, path in enumerate(paths):
        line, = ax.plot([], [], marker='o', color=color_cycle[i % 5], linewidth=2, label=f'Task {i+1}')
        lines.append(line)
    # 전환점
    switch_xs = [full_path[idx][1] for idx in task_splits[1:-1]]
    switch_ys = [full_path[idx][0] for idx in task_splits[1:-1]]
    ax.scatter(switch_xs, switch_ys, color='orange', s=60, marker='^', label='Task Switch')

    def init():
        for l in lines:
            l.set_data([], [])
        return lines

    def animate(frame):
        for i, (start, end) in enumerate(zip(seg_cumsum[:-1], seg_cumsum[1:])):
            length = end - start
            show_to = min(frame, length-1)
            if show_to < 0:
                lines[i].set_data([], [])
            else:
                path = full_path[start:start+show_to+1]
                if path:
                    ys, xs = zip(*path)
                    lines[i].set_data(xs, ys)
        return lines

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=max(seg_lengths), interval=interval, blit=True, repeat=False)
    ax.legend()
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow')
        else:
            ani.save(save_path, writer='ffmpeg')
    plt.show()


# # 전체 경로와 충돌 한 번에 보기
# plot_map_with_paths(my_map, start_locs, goal_locs, paths, obstacles, racks, picking_zones, collisions)
# # agent 0번만 단독 분석
# plot_single_path(my_map, paths[0], agent_idx=0)
# # 애니메이션 생성 및 저장 / 주피터에선 interactivity/ipython.display
# animate_paths(my_map, paths, obstacles, racks, picking_zones, interval=300, save_path="mapf_sim.gif")

def plot_multi_paths(my_map, paths, starts=None, goals=None, racks=None,
                     title='CBS Batch Paths', save_path=None,
                     agent_marker_size=6, agent_line_width=1.2, rack_alpha=0.25):
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.imshow(my_map, cmap='Greys', origin='lower')

    if racks:
        ys, xs = zip(*racks)
        ax.scatter(xs, ys, c='blue', marker='s', s=80, alpha=rack_alpha, label='Rack')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, p in enumerate(paths):
        if not p: 
            continue
        ys, xs = zip(*p)
        steps = len(p)                     # 길이 자동 계산
        c = colors[i % len(colors)]
        lbl = f'AGV{i} ({steps} steps)'    # 범례에 steps 표시
        ax.plot(xs, ys, '-', linewidth=agent_line_width, color=c, label=lbl)
        ax.scatter([xs[0]], [ys[0]], s=50, marker='s', color=c)
        ax.scatter([xs[-1]], [ys[-1]], s=60, marker='*', color=c)
        ax.text(xs[-1]+0.3, ys[-1]+0.3, f'{steps}', color=c, fontsize=8)  # 골 근처 텍스트

    ax.set_title(title)
    ax.legend(fontsize=8, loc='upper right')
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def animate_multi_paths(my_map, paths, interval=80, title='CBS Batch Animation',
                        save_path=None, agent_marker_size=6, agent_line_width=1.2):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(my_map, cmap='Greys', origin='lower')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    maxT = max(len(p) for p in paths if p)
    lines, heads = [], []

    # 범례에 steps 넣기
    for i, p in enumerate(paths):
        c = colors[i % len(colors)]
        steps = len(p) if p else 0
        label = f'AGV{i} ({steps} steps)'
        line, = ax.plot([], [], '-', linewidth=agent_line_width, color=c, label=label)
        head = ax.scatter([], [], s=agent_marker_size*5, marker='o', color=c)
        lines.append(line); heads.append(head)

    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    def frame(t):
        for i, p in enumerate(paths):
            if not p: 
                continue
            tt = min(t, len(p)-1)
            ys, xs = zip(*p[:tt+1])
            lines[i].set_data(xs, ys)
            heads[i].set_offsets([[xs[-1], ys[-1]]])
        return lines + heads

    ani = animation.FuncAnimation(fig, frame, frames=maxT, interval=interval, blit=False, repeat=False)
    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer='pillow')
        else:
            ani.save(save_path, writer='ffmpeg')
    plt.close()
