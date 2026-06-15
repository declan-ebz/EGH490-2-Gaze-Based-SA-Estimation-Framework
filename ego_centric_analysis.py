"""
ego_centric_analysis.py
Master framework execution pipeline for Ego-Centric Driving Scene Understanding.
"""
import os
import sys
import re
import json
import glob
import cv2
import numpy as np
import textwrap
from tqdm import tqdm
from PIL import Image
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from ultralytics import YOLO
import google.generativeai as genai
from google.colab import userdata

import config
import utils
from models import TemporalObjectTracker, PersistentKnowledgeGraph

# Setup paths for local SAM2 structures
if config.SAM2_REPO_ROOT not in sys.path:
    sys.path.insert(0, config.SAM2_REPO_ROOT)
from sam2.build_sam import build_sam2_video_predictor

def main():
    # --- Initialize API Keys & Engines ---
    print("Initializing API interfaces and tracking logic...")
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.LOCAL_SCENE_DIR, exist_ok=True)
    os.makedirs(config.VISUAL_DIR, exist_ok=True)

    # --- Preprocessing & Caching ---
    print("Structuring video sequence paths...")
    raw_files = sorted([os.path.join(root, f) for root, _, files in os.walk(config.DRIVE_SCENE_DIR) for f in files if f.lower().endswith(('.png', '.jpg'))])
    for i, filepath in enumerate(raw_files):
        target_path = os.path.join(config.LOCAL_SCENE_DIR, f"{i:05d}.jpg")
        if not os.path.exists(target_path):
            import shutil
            shutil.copy(filepath, target_path)

    frame_names = sorted(os.listdir(config.LOCAL_SCENE_DIR))
    total_frames = len(frame_names)

    # --- Eye Gaze Initialization ---
    print("Processing eye tracking datasets...")
    gaze_dir = os.path.join(config.DATA_DIR, 'gaze_info')
    depth_dir = os.path.join(config.DATA_DIR, 'scene_depth')
    
    fixations = utils.detect_fixations(gaze_dir)
    mapped_results = []
    for fix in fixations:
        frame_str = f"{fix.frame_start:08d}"
        scene_path = os.path.join(config.DRIVE_SCENE_DIR, f"{frame_str}_scene.png")
        depth_path = os.path.join(depth_dir, f"{frame_str}_depth.npy")
        mapped_xy, world_pt = utils.map_fixation_to_scene([fix.x, fix.y], scene_path, depth_path)
        if mapped_xy is not None:
            mapped_results.append({'frame_start': fix.frame_start, 'mapped_2d': mapped_xy, 'world_point': world_pt, 'duration_ms': fix.duration_ms})

    gaze_prompts = utils.generate_gaze_prompts(mapped_results)

    # --- Load Models ---
    print("Loading deep-learning perception networks...")
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.VLM_MODEL_NAME,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto"
    )
    vlm_processor = AutoProcessor.from_pretrained(config.VLM_MODEL_NAME, trust_remote_code=True)
    yolo_model = YOLO(config.YOLO_MODEL_PATH)
    tracker = TemporalObjectTracker()

    idx_to_orig_id = {i: str(int(re.findall(r'\d+', fname)[0])) for i, fname in enumerate(raw_files) if re.findall(r'\d+', fname)}
    all_frame_results = []

    # --- Semantic Logic Extraction Pass ---
    print("Running multi-stage semantic feature triage loops...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, name in enumerate(tqdm(frame_names)):
            img_bgr = cv2.imread(os.path.join(config.LOCAL_SCENE_DIR, name))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            res = yolo_model.predict(img_rgb, verbose=False, conf=0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            matched, _ = tracker.match_objects(boxes, [str(c) for c in classes], res.boxes.conf.cpu().numpy(), classes)
            track_ids = [matched[i] for i in range(len(boxes))]

            current_vlm_dict, current_vlm_primary, current_vlm_secondary = {}, {}, []

            if idx % 10 == 0 and len(boxes) > 0:
                vlm_img = img_rgb.copy()
                for i, box in enumerate(boxes):
                    cv2.putText(vlm_img, f"T{track_ids[i]}", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                base_image = Image.fromarray(vlm_img)

                # Prompt Triage Layer
                prompt_triage = "Classify important objects into ONE of three categories: imminent, latent, poi. Respond ONLY with comma-separated pairs like T0: imminent."
                msg_triage = [{"role": "user", "content": [{"type": "image", "image": base_image}, {"type": "text", "text": prompt_triage}]}]
                text_triage = vlm_processor.apply_chat_template(msg_triage, tokenize=False, add_generation_prompt=True)
                out_triage = vlm_processor.batch_decode(vlm_model.generate(**vlm_processor(text=[text_triage], images=[base_image], padding=True, return_tensors="pt").to("cuda"), max_new_tokens=50), skip_special_tokens=True)[0]

                for match in re.findall(r'(T\d+)[\s:-]*(imminent|latent|poi)', out_triage, re.IGNORECASE):
                    current_vlm_dict[match[0].upper()] = match[1].lower()

                current_valid_tids = [f"T{t}" for t in track_ids]
                important_tids = [tid for tid in current_vlm_dict.keys() if tid in current_valid_tids]

                if important_tids:
                    tids_str = ", ".join(important_tids)
                    # Ego -> Object
                    prompt_primary = f"Look at these specific tracked objects: {tids_str}. Describe location relative to EGO car in under 5 words per object. Format: REL: EGO->T0: parked on right"
                    msg_prim = [{"role": "user", "content": [{"type": "image", "image": base_image}, {"type": "text", "text": prompt_primary}]}]
                    text_prim = vlm_processor.apply_chat_template(msg_prim, tokenize=False, add_generation_prompt=True)
                    out_prim = vlm_processor.batch_decode(vlm_model.generate(**vlm_processor(text=[text_prim], images=[base_image], padding=True, return_tensors="pt").to("cuda"), max_new_tokens=150), skip_special_tokens=True)[0]

                    for tid, rel in re.findall(r'REL:\s*EGO\s*->\s*(T\d+)[\s:-]+([^\n]+)', out_prim, re.IGNORECASE):
                        if tid.upper() in important_tids:
                            current_vlm_primary[tid.upper()] = rel.strip().lower()

                    # Object -> Object
                    prompt_sec = f"Look at these tracked objects: {tids_str}. Identify if any are near each other and describe relationship. Format: REL: T2->T3: following behind"
                    msg_sec = [{"role": "user", "content": [{"type": "image", "image": base_image}, {"type": "text", "text": prompt_sec}]}]
                    text_sec = vlm_processor.apply_chat_template(msg_sec, tokenize=False, add_generation_prompt=True)
                    out_sec = vlm_processor.batch_decode(vlm_model.generate(**vlm_processor(text=[text_sec], images=[base_image], padding=True, return_tensors="pt").to("cuda"), max_new_tokens=150), skip_special_tokens=True)[0]

                    for src, tgt, rel_text in re.findall(r'REL:\s*(T\d+)\s*->\s*(T\d+)[\s:-]+([^\n]+)', out_sec, re.IGNORECASE):
                        if src.upper() in important_tids and tgt.upper() in important_tids:
                            current_vlm_secondary.append((src.upper(), rel_text.strip().lower(), tgt.upper()))

            orig_frame_id = idx_to_orig_id.get(idx, str(idx))
            raw_gaze_data = gaze_prompts.get(orig_frame_id, [])
            current_gaze = [pt['coords'] for pt in raw_gaze_data if 'coords' in pt]

            all_frame_results.append({
                'boxes': boxes, 'track_ids': track_ids, 'classes': classes, 'img': img_rgb, 'gaze_points': current_gaze,
                'vlm_dict': current_vlm_dict.copy(), 'vlm_primary': current_vlm_primary.copy(), 'vlm_secondary': list(current_vlm_secondary)
            })

    yolo_names = yolo_model.names
    del vlm_model, yolo_model
    utils.purge_vram()

    # --- Temporal Mask Tracking Propagation Layer ---
    print("Seeding SAM2 spatial mask memory lines...")
    video_predictor = build_sam2_video_predictor(config.SAM2_MODEL_CFG, config.SAM2_CHECKPOINT, device=config.DEVICE)
    inference_state = video_predictor.init_state(video_path=config.LOCAL_SCENE_DIR, offload_video_to_cpu=True, offload_state_to_cpu=True)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for idx, data in enumerate(all_frame_results):
            for i, box in enumerate(data['boxes'][:10]):
                video_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=idx, obj_id=data['track_ids'][i], box=box)
            for i, pt in enumerate(data['gaze_points']):
                video_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=idx, obj_id=100 + i, points=np.array([pt], dtype=np.float32), labels=np.array([1], dtype=np.int32))

        video_masks_cache = {}
        for out_idx, out_ids, out_logits in video_predictor.propagate_in_video(inference_state):
            video_masks_cache[out_idx] = {'ids': out_ids, 'masks': (out_logits > 0.0).cpu().numpy()}

    video_predictor.reset_state(inference_state)
    del video_predictor
    utils.purge_vram()

    # --- Spatiotemporal Knowledge Graph Updates & Evaluation Summary ---
    print("Compiling global scene maps and downstream metrics...")
    knowledge_graph = PersistentKnowledgeGraph()
    temporal_lifecycle = {}
    all_gemini_advice_data = []

    for idx in tqdm(range(total_frames)):
        data = all_frame_results[idx]
        current_gaze = data.get('gaze_points', [])

        fixated_ids = set()
        for pt in current_gaze:
            px, py = pt[0], pt[1]
            for i, box in enumerate(data['boxes']):
                closest_x = max(box[0], min(px, box[2]))
                closest_y = max(box[1], min(py, box[3]))
                if np.sqrt((px - closest_x)**2 + (py - closest_y)**2) <= config.FOVEAL_RADIUS:
                    fixated_ids.add(f"T{data['track_ids'][i]}")

        current_nodes = [{'id': 'EGO', 'label': 'Driver'}]
        for i, tid in enumerate(data['track_ids']):
            current_nodes.append({'id': f"T{tid}", 'label': yolo_names[data['classes'][i]]})

        knowledge_graph.update_from_frame(idx, current_nodes, fixated_ids, data.get('vlm_dict', {}), data.get('vlm_primary', {}), data.get('vlm_secondary', []))
        graph_dict = knowledge_graph.to_dict()

        for node in graph_dict['nodes']:
            tid = node['id']
            if tid not in temporal_lifecycle: temporal_lifecycle[tid] = [None] * total_frames
            temporal_lifecycle[tid][idx] = node.get('status')

        # Multi-stage Prompt Filtering for Advice Generation
        candidate_advices = []
        for i in range(3):
            try:
                instructor_prompt = textwrap.dedent(f"""
                    You are an experienced driving instructor. Provide direct, single-sentence advice under 25 words to a novice driver.
                    Highlight key tactical priorities or hidden threats. Format output strictly as JSON:
                    {{ "frame_id": "{idx}", "gemini_advice": "Advice text here." }}
                """)
                response = gemini_model.generate_content([instructor_prompt, Image.fromarray(data['img'])], generation_config={"response_mime_type": "application/json"})
                candidate_advices.append(json.loads(response.text).get('gemini_advice', ''))
            except Exception:
                candidate_advices.append('')

        filtered_candidates = [adv for adv in candidate_advices if adv]
        gemini_advice_for_frame = "No advice could be generated."
        if filtered_candidates:
            try:
                selection_prompt = textwrap.dedent(f"""
                    You are a senior driving instructor. Review these options and return the single most actionable alert under 25 words.
                    Format output strictly as JSON. Options:\n""" + "\n".join(filtered_candidates))
                sel_resp = gemini_model.generate_content([selection_prompt, Image.fromarray(data['img'])], generation_config={"response_mime_type": "application/json"})
                gemini_advice_for_frame = json.loads(sel_resp.text).get('gemini_advice', 'Could not select best advice.')
            except Exception:
                pass

        all_gemini_advice_data.append({"frame_id": str(idx), "gemini_advice": gemini_advice_for_frame})
        with open(config.GEMINI_ADVICE_JSON_PATH, 'w') as f:
            json.dump(all_gemini_advice_data, f, indent=4)

        # Situational Awareness Evaluation Summary scoring
        situational_awareness_score = "N/A"
        try:
            sa_prompt = textwrap.dedent(f"""
                Rate the operator's current situational awareness level as Low, Medium, or High based on the scene layout and gaze points.
                Graph details: {json.dumps(graph_dict)}. Output format strictly JSON:
                {{ "frame_id": "{idx}", "situational_awareness": "Medium" }}
            """)
            sa_resp = gemini_model.generate_content([sa_prompt, Image.fromarray(data['img'])], generation_config={"response_mime_type": "application/json"})
            situational_awareness_score = json.loads(sa_resp.text).get('situational_awareness', 'N/A')
        except Exception:
            pass

        # Parse cross-references and suppress unhelpful interface flags
        display_gemini_advice = False
        advice_lower = gemini_advice_for_frame.lower()
        if re.search(r'\b(hazard|attention|missed|warning|risk|avoid|watch)\b', advice_lower):
            display_gemini_advice = True

        utils.visualize_ego_centric_rolling(
            idx, data['img'], data['boxes'], current_gaze, current_nodes[1:], graph_dict, temporal_lifecycle,
            gemini_advice_for_frame, situational_awareness_score, display_gemini_advice,
            os.path.join(config.VISUAL_DIR, f"eval_map_{idx:05d}.jpg")
        )

    # --- Render Video Clip ---
    print("Compiling system execution output video file...")
    image_files = sorted(glob.glob(os.path.join(config.VISUAL_DIR, "eval_map_*.jpg")))
    if image_files:
        first_frame = cv2.imread(image_files[0])
        h, w, _ = first_frame.shape
        video_writer = cv2.VideoWriter(config.VIDEO_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
        for img_path in image_files:
            video_writer.write(cv2.imread(img_path))
        video_writer.release()
        print(f"Pipeline complete. Video file generated successfully: {config.VIDEO_OUTPUT_PATH}")

if __name__ == "__main__":
    main()