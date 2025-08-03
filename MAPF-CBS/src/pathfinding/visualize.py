# visualize.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import itertools

def plot_map_with_paths(
        my_map, start_locs, goal_locs, paths,
        obstacles=None, racks=None, picking_zones=None, collisions=None,
        agent_names=None, title='MAPF Paths', save_path=None):
    """
    - my_map: 2D array (0: free, 1: obstacle, ...)
    - start_locs/goal_locs: list of (y, x)
    - paths: list of [ (y, x), ... ] per agent
    - obstacles/racks/picking_zones: list of (y, x) (optional)
    - collisions: list of dict (optional)
    - agent_names: list of str (optional)
    - title, save_path: 그림 제목, 저장 경로
    """
    # 1. 맵, 장애물, 랙, 피킹존, 배경 그리기
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap("Greys")
    plt.imshow(my_map, cmap=cmap, origin='upper')

    # 장애물, 랙, 피킹존 등 오버레이
    if racks:
        ys, xs = zip(*racks)
        plt.scatter(xs, ys, c='blue', marker='s', s=140, label='Rack')
    if picking_zones:
        ys, xs = zip(*picking_zones)
        plt.scatter(xs, ys, c='purple', marker='P', s=120, label='Picking Zone')

    # agent별 path 시각화
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, path in enumerate(paths):
        ys, xs = zip(*path)
        name = agent_names[idx] if agent_names else f'Agent {idx}'
        plt.plot(xs, ys, marker='o', label=name, color=color_cycle[idx % len(color_cycle)], linewidth=2)
        plt.scatter([xs[0]], [ys[0]], color=color_cycle[idx % len(color_cycle)], s=100, marker='s')
        plt.scatter([xs[-1]], [ys[-1]], color=color_cycle[idx % len(color_cycle)], s=100, marker='*')

    # 충돌 표시
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
    """
    - path: [(y, x), ...]
    - agent_idx: 색상 인덱스
    """
    plt.figure(figsize=(6,6))
    cmap = plt.get_cmap("Greys")
    plt.imshow(my_map, cmap=cmap, origin='upper')
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ys, xs = zip(*path)
    plt.plot(xs, ys, marker='o', color=color_cycle[agent_idx % len(color_cycle)], linewidth=2)
    plt.scatter([xs[0]], [ys[0]], color=color_cycle[agent_idx % len(color_cycle)], s=120, marker='s')
    plt.scatter([xs[-1]], [ys[-1]], color=color_cycle[agent_idx % len(color_cycle)], s=120, marker='*')
    plt.title(title or f"Agent {agent_idx} Path")
    plt.gca().invert_yaxis()
    plt.show()

def animate_paths(
        my_map, paths, obstacles=None, racks=None, picking_zones=None,
        interval=500, title='MAPF Animation', save_path=None):
    """
    - paths: list of [(y, x), ...] per agent
    - interval: frame(ms)
    - save_path: None, "animation.gif" 또는 "animation.mp4"
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    max_t = max(len(p) for p in paths)

    fig, ax = plt.subplots(figsize=(8,8))
    cmap = plt.get_cmap("Greys")
    ax.imshow(my_map, cmap=cmap, origin='upper')

    scatters = []
    for i, path in enumerate(paths):
        scatter, = ax.plot([], [], marker='o', color=color_cycle[i % len(color_cycle)], linewidth=2, label=f'Agent {i}')
        scatters.append(scatter)

    def init():
        for s in scatters:
            s.set_data([], [])
        return scatters

    def animate(frame):
        for i, path in enumerate(paths):
            t = min(frame, len(path)-1)
            ys, xs = zip(*path[:t+1])
            scatters[i].set_data(xs, ys)
        return scatters

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=max_t, interval=interval, blit=True, repeat=False)
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


from visualize import plot_map_with_paths, plot_single_path, animate_paths
# 전체 경로와 충돌 한 번에 보기
plot_map_with_paths(my_map, start_locs, goal_locs, paths, obstacles, racks, picking_zones, collisions)
# agent 0번만 단독 분석
plot_single_path(my_map, paths[0], agent_idx=0)
# 애니메이션 생성 및 저장 / 주피터에선 interactivity/ipython.display
animate_paths(my_map, paths, obstacles, racks, picking_zones, interval=300, save_path="mapf_sim.gif")
