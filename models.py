"""
models.py
Defines the TemporalObjectTracker and PersistentKnowledgeGraph data models.
"""
import networkx as nx
import config

class TemporalObjectTracker:
    """Computes IoU associations to preserve object persistence across video tracking frames."""
    def __init__(self, window_size=config.GHOST_LIMIT, iou_threshold=0.3, confidence_threshold=0.5):
        self.window_size = window_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.tracked_objects = {}
        self.next_track_id = 0
        self.frame_history = []

    def compute_iou(self, box1, box2):
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        if x2 < x1 or y2 < y1: return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter)

    def match_objects(self, current_boxes, current_labels, current_confidences, yolo_classes):
        matched_tracks, unmatched_detections = {}, list(range(len(current_boxes)))
        for track_id, track_data in list(self.tracked_objects.items()):
            best_iou, best_match = 0, None
            for det_idx in unmatched_detections:
                iou = self.compute_iou(track_data['box'], current_boxes[det_idx])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_match = iou, det_idx
            if best_match is not None:
                matched_tracks[best_match] = track_id
                unmatched_detections.remove(best_match)
                track_data.update({'box': current_boxes[best_match], 'label': current_labels[best_match], 'confidence': current_confidences[best_match]})
        for det_idx in unmatched_detections:
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracked_objects[tid] = {'box': current_boxes[det_idx], 'label': current_labels[det_idx], 'confidence': current_confidences[det_idx], 'history': []}
            matched_tracks[det_idx] = tid
        return matched_tracks, []

class PersistentKnowledgeGraph:
    """Manages active node memory updates and handles structural state logic loops."""
    def __init__(self, ghost_limit=config.GHOST_LIMIT, memory_window=config.MEMORY_WINDOW):
        self.graph = nx.MultiDiGraph()
        self.last_seen = {}
        self.last_fixated = {}
        self.primary_memory = {}
        self.secondary_memory = {}
        self.ghost_limit = ghost_limit
        self.memory_window = memory_window
        self.graph.add_node('EGO', label='Driver', status='ego')

    def update_from_frame(self, frame_idx, current_objs, fixated_ids, vlm_dict, vlm_primary, vlm_secondary):
        visible_ids = set()
        for obj in current_objs:
            node_id = obj['id']
            if node_id == 'EGO': continue
            visible_ids.add(node_id)
            self.last_seen[node_id] = frame_idx

            is_fixated = node_id in fixated_ids
            if is_fixated:
                self.last_fixated[node_id] = frame_idx

            time_since_fixation = frame_idx - self.last_fixated.get(node_id, -999)
            is_remembered = (0 < time_since_fixation <= self.memory_window)

            existing_class = self.graph.nodes[node_id].get('vlm_class', '') if node_id in self.graph.nodes else ""
            vlm_class = vlm_dict.get(node_id, existing_class)
            base_type = vlm_class if vlm_class else "normal"

            if is_fixated: status = f"{base_type}_active"
            elif is_remembered: status = f"{base_type}_remembered"
            else: status = f"{base_type}_unseen"

            self.graph.add_node(node_id, label=obj['label'], status=status, vlm_class=vlm_class)

        for tid, rel in vlm_primary.items():
            self.primary_memory[tid] = rel
        for src, rel, tgt in vlm_secondary:
            self.secondary_memory[(src, tgt)] = rel

        all_nodes = list(self.graph.nodes)
        for node_id in all_nodes:
            if node_id == 'EGO': continue
            if node_id not in visible_ids:
                age = frame_idx - self.last_seen.get(node_id, 0)
                if age <= self.ghost_limit:
                    if 'active' in self.graph.nodes[node_id].get('status', '') or 'remembered' in self.graph.nodes[node_id].get('status', ''):
                        base = self.graph.nodes[node_id].get('vlm_class', '') or 'normal'
                        self.graph.nodes[node_id]['status'] = f"{base}_remembered"
                    else:
                        self.graph.nodes[node_id]['status'] = 'ghost'
                else:
                    self.graph.remove_node(node_id)
                    if node_id in self.last_fixated: del self.last_fixated[node_id]

        active_nodes = set(self.graph.nodes)
        keys_to_remove = [k for k in self.secondary_memory.keys() if k[0] not in active_nodes or k[1] not in active_nodes]
        for k in keys_to_remove: del self.secondary_memory[k]

        self.graph.remove_edges_from(list(self.graph.edges))

        for node_id in self.graph.nodes:
            if node_id == 'EGO': continue
            status = self.graph.nodes[node_id].get('status', 'normal_unseen')

            if 'active' in status: cog_state = 'LOOKING'
            elif 'remembered' in status: cog_state = 'REMEMBERED'
            elif 'unseen' in status and 'normal' not in status: cog_state = 'MISSED'
            elif 'ghost' in status: cog_state = 'GHOST'
            else: cog_state = 'OBSERVED'

            if cog_state == 'GHOST':
                final_relation = "[GHOST]"
            else:
                rel_text = self.primary_memory.get(node_id, "")
                final_relation = f"{rel_text.upper()}\n[{cog_state}]" if rel_text else f"[{cog_state}]"

            self.graph.add_edge('EGO', node_id, relation=final_relation, edge_type='primary')

        for (src, tgt), rel_text in self.secondary_memory.items():
            if src in self.graph.nodes and tgt in self.graph.nodes:
                status_src = self.graph.nodes[src].get('status', '')
                status_tgt = self.graph.nodes[tgt].get('status', '')
                if 'ghost' not in status_src and 'ghost' not in status_tgt:
                    self.graph.add_edge(src, tgt, relation=rel_text.upper(), edge_type='secondary')

    def to_dict(self):
        return {'nodes': [dict(self.graph.nodes[n], id=n) for n in self.graph.nodes],
                'edges': [dict(self.graph.edges[u, v, k], **{'from': u, 'to': v, 'edge_type': d.get('edge_type', 'primary'), 'relation': d.get('relation', '')})
                          for u, v, k, d in self.graph.edges(keys=True, data=True)]}