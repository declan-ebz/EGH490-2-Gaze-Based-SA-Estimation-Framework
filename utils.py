"""
utils.py
Core utility functions for gaze telemetry parsing, fixation detection,
matrix mappings, visualization, and VRAM management.
"""
import os
import gc
import re
import json
import torch
import numpy as np
import cv2
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib as mpl
import networkx as nx
from collections import namedtuple, defaultdict
import config

# Named tuple configuration for structured fixation data
Fixation = namedtuple('Fixation', ['frame_start', 'frame_end', 'x', 'y', 'duration_ms', 'pupil_size'])

def purge_vram():
    """Flushes background cache layers to stabilize consumer edge GPUs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def parse_gaze_file(gaze_path, frame_rate=config.FRAME_RATE):
    """Parses raw, unformatted eye-tracking output files safely."""
    with open(gaze_path, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(': ', 1)
            cleaned_value = value.strip('[]').replace(',', ' ').strip()
            try:
                data[key] = np.fromstring(cleaned_value, sep=' ')
            except ValueError:
                data[key] = np.array([])
                
    gaze_2d = data.get('Gaze_Loc_2D', np.array([0.0, 0.0]))
    if len(gaze_2d) < 2:
        gaze_2d = np.array([0.0, 0.0])
    pupil_size = None  
    frame_idx = int(os.path.basename(gaze_path).split('_')[0])
    return frame_idx, float(gaze_2d[0]), float(gaze_2d[1]), pupil_size

def detect_fixations(gaze_dir, dispersion_threshold=config.DISPERSION_THRESHOLD, 
                      min_duration_ms=config.MIN_DURATION_MS, frame_rate=config.FRAME_RATE):
    """Groups scattered gaze coordinate trends into spatial fixations."""
    if not os.path.exists(gaze_dir):
        print(f"Warning: Directory {gaze_dir} not found. Skipping.")
        return []

    gaze_files = sorted([f for f in os.listdir(gaze_dir) if f.endswith('_gaze.txt')])
    if not gaze_files:
        print(f"Warning: No gaze files found in {gaze_dir}.")
        return []

    gaze_points = []
    for f in gaze_files:
        gaze_path = os.path.join(gaze_dir, f)
        try:
            frame_idx, x, y, pupil = parse_gaze_file(gaze_path, frame_rate)
            gaze_points.append((frame_idx, x, y, pupil))
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            continue

    if not gaze_points:
        return []

    fixations = []
    current_cluster = []
    for point in gaze_points:
        if not current_cluster:
            current_cluster.append(point)
            continue
        cluster_mean = np.mean([p[1:3] for p in current_cluster], axis=0)
        dist = np.linalg.norm(np.array(point[1:3]) - cluster_mean)
        if dist <= dispersion_threshold:
            current_cluster.append(point)
        else:
            num_frames = len(current_cluster)
            duration_ms = (num_frames / frame_rate) * 1000
            if duration_ms >= min_duration_ms:
                fix_mean = np.mean([p[1:3] for p in current_cluster], axis=0)
                pupil_avg = np.mean([p[3] for p in current_cluster if p[3] is not None]) if any(p[3] is not None for p in current_cluster) else None
                fixations.append(Fixation(current_cluster[0][0], current_cluster[-1][0], fix_mean[0], fix_mean[1], duration_ms, pupil_avg))
            current_cluster = [point]

    if current_cluster:
        num_frames = len(current_cluster)
        duration_ms = (num_frames / frame_rate) * 1000
        if duration_ms >= min_duration_ms:
            fix_mean = np.mean([p[1:3] for p in current_cluster], axis=0)
            pupil_avg = np.mean([p[3] for p in current_cluster if p[3] is not None]) if any(p[3] is not None for p in current_cluster) else None
            fixations.append(Fixation(current_cluster[0][0], current_cluster[-1][0], fix_mean[0], fix_mean[1], duration_ms, pupil_avg))

    return fixations

def map_fixation_to_scene(fixation_xy, scene_path, depth_path=None, homography_matrix=None, intrinsics=config.INTRINSICS):
    """Maps coordinates from eye-tracking devices onto spatial camera frames."""
    if not os.path.exists(scene_path):
        return None, None

    scene_img = cv2.imread(scene_path)
    if scene_img is None:
        return None, None
    scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    height, width = scene_img.shape[:2]

    norm_xy = np.array(fixation_xy) / np.array([width, height])
    scaled_xy = norm_xy * np.array([width, height])

    if homography_matrix is not None:
        pt = np.array([[scaled_xy[0], scaled_xy[1], 1]]).T
        transformed_pt = homography_matrix @ pt
        scaled_xy = transformed_pt[:2].flatten() / transformed_pt[2]

    world_point = None
    if depth_path and os.path.exists(depth_path) and intrinsics:
        depth_map = np.load(depth_path)
        if depth_map.shape[:2] == (height, width):
            y_idx, x_idx = int(scaled_xy[1]), int(scaled_xy[0])
            if 0 <= y_idx < height and 0 <= x_idx < width:
                depth_value = depth_map[y_idx, x_idx]
                cx, cy, fx, fy = intrinsics
                world_point = depth_value * np.array([(scaled_xy[0] - cx) / fx, (scaled_xy[1] - cy) / fy, 1])

    return scaled_xy, world_point

def generate_gaze_prompts(mapped_results, merge_distance=30, base_radius=20, output_path=config.GAZE_PROMPTS_PATH):
    """Formats processed fixations into bounding cues structured for SAM2 tracking entry points."""
    frame_prompts = defaultdict(list)
    for fix in mapped_results:
        frame_id = fix['frame_start']
        x, y = fix['mapped_2d']
        duration = fix.get('duration_ms', 0)
        frame_prompts[frame_id].append({'coords': np.array([x, y]), 'duration_ms': duration})

    final_prompts = {}
    for frame_id, fixations in frame_prompts.items():
        merged = []
        used = np.zeros(len(fixations), dtype=bool)
        for i, fix in enumerate(fixations):
            if used[i]: continue
            cluster = [fix]
            for j, other in enumerate(fixations):
                if i != j and not used[j]:
                    dist = np.linalg.norm(fix['coords'] - other['coords'])
                    if dist < merge_distance:
                        cluster.append(other)
                        used[j] = True

            coords = np.mean([f['coords'] for f in cluster], axis=0)
            total_duration = sum(f['duration_ms'] for f in cluster)
            radius = base_radius + (total_duration / 100.0)

            merged.append({
                'type': 'circle',
                'coords': (float(coords[0]), float(coords[1])),
                'radius': float(radius),
                'duration_ms': total_duration
            })
        final_prompts[frame_id] = merged

    with open(output_path, 'w') as f:
        json.dump(final_prompts, f, indent=2)
    return final_prompts

def visualize_ego_centric_rolling(frame_id, img_rgb, boxes, gaze_pts, labeled_data, graph_data, lifecycle, 
                                  gemini_advice_for_frame, situational_awareness_score, display_advice_flag, output_path):
    """Compiles ego-centric tracking overlays, NetworkX knowledge graphs, and memory Gantt charts into single frames."""
    fig = plt.figure(figsize=(26, 16), facecolor='#f8f9fa')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 0.4])

    # --- TOP LEFT: REAL-TIME OVERLAY CAMERA VIEW ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    colors_map = mpl.colormaps.get_cmap('hsv')

    for i, box in enumerate(boxes):
        tid = labeled_data[i]['id']
        node_data = next((n for n in graph_data['nodes'] if n['id'] == tid), {})
        vtype = node_data.get('vlm_class', '')
        status = node_data.get('status', '')
        track_num = int(tid.replace('T','')) if 'T' in tid else 0
        
        if vtype == 'imminent': base_color = '#e74c3c'
        elif vtype == 'latent': base_color = '#e67e22'
        elif vtype == 'poi': base_color = '#9b59b6'
        else: base_color = colors_map((track_num % 20) / 20)

        is_active = 'active' in status
        is_remembered = 'remembered' in status
        box_edge = '#2ecc71' if is_active else base_color
        linewidth = 5 if (vtype or is_active or is_remembered) else 3

        ax1.add_patch(patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                         linewidth=linewidth, edgecolor=box_edge, facecolor='none', zorder=2))

        label_text = tid
        if vtype: label_text += f" [{vtype.upper()}]"
        if is_active: 
            label_text += " (LOOKING)"
            label_bg = '#2ecc71'
        elif is_remembered: 
            label_text += " (REMEMBERED)"
            label_bg = base_color
        else: 
            label_bg = base_color

        ax1.text(box[0], box[1]-10, label_text, color='white', fontweight='bold', zorder=3,
                bbox=dict(facecolor=label_bg, alpha=0.9, edgecolor='white' if is_remembered else 'none', linewidth=2 if is_remembered else 0, boxstyle='round,pad=0.3'))

    for pt in gaze_pts:
        ax1.scatter(pt[0], pt[1], s=1000, c='#2ecc71', marker='o', alpha=0.4, edgecolors='none', zorder=4)
        ax1.scatter(pt[0], pt[1], s=150, c='#e74c3c', marker='x', linewidths=4, zorder=5)

    if display_advice_flag and gemini_advice_for_frame:
        wrapped_advice = textwrap.fill(gemini_advice_for_frame, width=50)
        ax1.text(0.02, 0.98, f"Gemini Advice:\n{wrapped_advice}", transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6))

    if situational_awareness_score:
        ax1.text(0.02, 0.85, f"Situational Awareness: {situational_awareness_score}", transform=ax1.transAxes, fontsize=12, verticalalignment='top', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7))

    legend_elements = [
        mlines.Line2D([], [], color='#e74c3c', marker='s', linestyle='None', markersize=12, label='Imminent Hazard'),
        mlines.Line2D([], [], color='#e67e22', marker='s', linestyle='None', markersize=12, label='Latent Hazard'),
        mlines.Line2D([], [], color='#9b59b6', marker='s', linestyle='None', markersize=12, label='Point of Interest (POI)'),
        mlines.Line2D([], [], color='#2ecc71', marker='o', linestyle='None', markersize=12, label='Active Fixation (Looking)'),
        mlines.Line2D([], [], markeredgecolor='#2ecc71', markerfacecolor='none', marker='o', markeredgewidth=4, linestyle='None', markersize=12, label='Working Memory (Remembered)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='white', framealpha=0.9)
    ax1.axis('off')

    # --- TOP RIGHT: SPATIOTEMPORAL GRAPH MODELLING ---
    ax2 = fig.add_subplot(gs[0, 1])
    G = nx.MultiDiGraph()
    node_colors, node_borders, node_linewidths, node_labels = [], [], [], {}
    base_colors = {'imminent': ('#e74c3c', '#c0392b'), 'latent': ('#e67e22', '#d35400'), 'poi': ('#9b59b6', '#8e44ad'), 'normal': ('#3498db', '#2980b9')}

    for n in graph_data['nodes']:
        node_id = n['id']
        G.add_node(node_id)
        label_str = f"{node_id}\n({n['label']})"
        if n.get('vlm_class'): label_str += f"\n[{n['vlm_class'].upper()}]"
        node_labels[node_id] = label_str

        status = n.get('status', '')
        vtype = n.get('vlm_class', 'normal') or 'normal'
        base_fill, base_border = base_colors.get(vtype, ('#3498db', '#2980b9'))

        if node_id == 'EGO':
            node_colors.append('#f1c40f'); node_borders.append('#333'); node_linewidths.append(3)
        else:
            if 'active' in status:
                node_colors.append('#2ecc71'); node_borders.append(base_fill); node_linewidths.append(4)
            elif 'remembered' in status:
                node_colors.append(base_fill); node_borders.append('#2ecc71'); node_linewidths.append(6)
            elif 'ghost' in status:
                node_colors.append('#bdc3c7'); node_borders.append('#95a5a6'); node_linewidths.append(2)
            else:
                node_colors.append(base_fill); node_borders.append(base_border); node_linewidths.append(3)

    pos = nx.spring_layout(G, k=3.5, seed=42)
    if 'EGO' in pos: pos['EGO'] = np.array([0, 0])

    edge_labels = {}
    for e in graph_data['edges']:
        u, v = e['from'], e['to']
        if u in pos and v in pos:
            rel_text = e.get('relation', '')
            status_v = next((n.get('status') for n in graph_data['nodes'] if n['id'] == v), 'in_frame')

            if 'active' in status_v: e_col, width = '#2ecc71', 4
            elif 'remembered' in status_v: e_col, width = '#27ae60', 3
            elif 'unseen' in status_v and 'normal' not in status_v:
                if 'imminent' in status_v: e_col, width = '#e74c3c', 3
                elif 'latent' in status_v: e_col, width = '#e67e22', 2
                elif 'poi' in status_v: e_col, width = '#9b59b6', 2
                else: e_col, width = '#bdc3c7', 1.5
            else: e_col, width = '#bdc3c7', 1.5

            nx.draw_networkx_edges(G, pos, edgelist=[(u,v)], ax=ax2, edge_color=e_col, width=width, arrowsize=20, connectionstyle='arc3,rad=0.1')
            if rel_text: edge_labels[(u,v)] = rel_text

    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, node_size=3500, edgecolors=node_borders, linewidths=node_linewidths)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax2, font_size=8, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2, font_size=7, font_color='#2c3e50', label_pos=0.6, bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=0.5))
    ax2.set_title("Fixation Mental Map", fontsize=20, fontweight='bold')
    ax2.axis('off')

    # --- BOTTOM: SYSTEM MEMORY TIMELINE ---
    ax3 = fig.add_subplot(gs[1, :])
    start_win = max(0, frame_id - config.ROLLING_WINDOW)
    end_win = max(config.ROLLING_WINDOW, frame_id + 5)

    active_tracks = [tid for tid, hist in lifecycle.items() if tid != 'EGO' and any(x is not None for x in hist[start_win:frame_id+1])]
    active_tracks = sorted(active_tracks, key=lambda x: int(x.replace('T','')) if 'T' in x else 0)

    for y_idx, tid in enumerate(active_tracks):
        history = lifecycle[tid][start_win : frame_id+1]
        for t_offset, status in enumerate(history):
            if status is None: continue
            if 'active' in status: bar_col = '#2ecc71'
            elif 'remembered' in status: bar_col = '#27ae60'
            elif 'unseen' in status:
                if 'imminent' in status: bar_col = '#e74c3c'
                elif 'latent' in status: bar_col = '#e67e22'
                elif 'poi' in status: bar_col = '#9b59b6'
                else: bar_col = '#3498db'
            else: bar_col = '#bdc3c7'
            ax3.barh(y_idx, 1, left=start_win + t_offset, color=bar_col, height=0.6)

    ax3.axvline(x=frame_id, color='#333333', linestyle='--', linewidth=2)
    ax3.set_xlim(start_win, end_win)
    ax3.set_yticks(range(len(active_tracks)))
    ax3.set_yticklabels(active_tracks, fontsize=8)
    ax3.set_title("Hazard Awareness Timeline (Green = Fixated/Memory)", fontsize=18, fontweight='bold')
    ax3.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()