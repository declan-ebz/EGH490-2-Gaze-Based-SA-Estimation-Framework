# Ego-Centric Driving Scene Understanding and Situational Awareness Framework

## Overview

This repository presents a novel framework for real-time ego-centric driving scene understanding, focusing on a driver's situational awareness. It integrates computer vision models (YOLOv8 for object detection, SAM2 for video object propagation) with visual language models (VLMs) and a persistent knowledge graph to analyze driver gaze, contextualize objects, and assess cognitive state. The system aims to simulate and evaluate a driver's attention and awareness in complex driving scenarios.

## Key Features

* **Gaze Fixation Detection**: Processes raw gaze data to identify and localize driver fixations within the scene.
* **Dynamic Object Tracking**: Utilizes YOLOv8 for object detection and a custom `TemporalObjectTracker` for maintaining object identities across video frames.
* **Semantic Scene Interpretation (VLM Integration)**: Employs a Visual Language Model (Qwen2.5-VL-7B-Instruct) to provide rich, human-like descriptions and relationships (EGO-centric and object-to-object) for critical scene elements, classifying them into 'imminent', 'latent', or 'point of interest' hazards.
* **SAM2-based Video Object Propagation**: Leverages SAM2 (Segment Anything Model 2) to propagate initial object detections and gaze prompts through video sequences, generating high-quality masks and temporal coherence.
* **Persistent Knowledge Graph**: A `PersistentKnowledgeGraph` maintains a dynamic mental model of the environment, tracking object states (active, remembered, ghosted) and VLM-derived relationships over time.
* **Situational Awareness Scoring**: A Gemini-powered evaluator assesses the driver's situational awareness based on object states, VLM insights, and gaze patterns.
* **Ego-Centric Visualization**: Generates rich visual summaries including camera view with object highlights, gaze overlay, a dynamic fixation mental map, and a rolling Gantt chart of object lifecycles and cognitive states.

---

## Repository Structure

* `ego_centric_analysis.ipynb`: The master Google Colab notebook containing the complete execution pipeline.
* `data_samples/`: Core directory containing verification assets:
* `gaze_info/`: Input eye-tracking coordinates and telemetry files (`*_gaze.txt`).
* `scene_ims/`: Input ego-centric driving video frame captures (`*_scene.png`).
* `scene_depth/`: Pre-computed scene depth maps (`*_depth.npy`).
* `face_ims/`: Input driver face tracking captures (`*_face.png`).


* `pretrained_models/`: Local cache directory storing model architecture checkpoints:
* `yolov8s.pt`: Pretrained object localization weights.
* `sam2.1_hiera_large.pt`: Target checkpoint for temporal mask propagation.



---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/declan-ebz/EGH490-2-Gaze-Based-SA-Estimation-Framework.git
cd EGH490-2-Gaze-Based-SA-Estimation-Framework

```

### 2. Google Colab Environment

This framework is optimized to run inside a Google Colab notebook environment. Ensure your active environment has a GPU runtime assigned (`Runtime > Change runtime type > GPU`).

> [!IMPORTANT]
> **Hardware Requirement**: This framework requires a GPU workspace with a minimum of **15GB VRAM** (e.g., Google Colab T4, L4, or A100 instances) to handle the combined memory footprint of YOLOv8, SAM2, and the quantized Qwen2.5-VL pipelines without encountering Out-Of-Memory (OOM) errors.

### 3. Verification Data and Model Checkpoints

All sample frame sequences and tracking coordinates are pre-packaged in the `data_samples/` folder.

* **YOLOv8s**: Pre-trained road detection weights are included directly at `pretrained_models/yolov8s.pt`.
* **SAM2**: The notebook provides a quick `wget` command cell to automatically pull down the heavy `sam2.1_hiera_large.pt` checkpoint straight into the `pretrained_models/` directory during setup.

### 4. Gemini API Key

To execute the multi-stage evaluation loops and generate driving instruction safety scores, you require a valid Google Gemini API Key. Store it securely in your Colab notebook secrets manager panel (the key icon on the left sidebar) under the specific token name: `GOOGLE_API_KEY`.

---

## Usage

1. **Open and Run the Notebook (`ego_centric_analysis.ipynb`)**:
Open the master file in Google Colab and run the cells sequentially to execute the full pipeline pipeline steps:
* Ingests system dependencies and sets up local workspace paths.
* Parses spatial data logs to map eye-gaze tracking frames.
* Runs YOLOv8 object localization paired with Qwen2.5-VL semantic hazard triage.
* Propagates mask boundaries over temporal sequences via SAM2 memory blocks.
* Populates and updates the active NetworkX `PersistentKnowledgeGraph` state tracking engine.
* Queries multi-stage Gemini safety loops to generate context-aware driving advice and driver situational awareness scores.
* Visualizes live framework states and compiles the final evaluation clip.


2. **Generated Outputs**:
The execution loop generates the following assets under your session folder:
* `gaze_prompts.json`: Processed coordinate gaze tracking points formatted for SAM2 consumption.
* `scene_segmentations/`: Workspace path containing frame overlay images (`eval_map_*.jpg`).
* `scene_segmentations/gemini_advice.json`: A structural database logging selected driver alerts and safety scores matched per frame index.
* `scene_segmentations/full_driving_scene_graph.mp4`: A compiled, high-definition video summary charting camera view targets, fixation network maps, and rolling memory timelines simultaneously.



---

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgements

* [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
* [Pre-trained Road Detection YOLOv8 Source Model](https://github.com/Mkoek213/road_detection_model.git)
* [Look Both Ways Driver Eye-Tracking Dataset](https://github.com/Kasai2020/look_both_ways)
* [Meta Segment Anything Model 2 (SAM2) Repository](https://github.com/facebookresearch/sam2)
* [Alibaba Qwen2.5-VL-7B-Instruct Core Language Model](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
* [Google Gemini API Developer Documentation](https://ai.google.dev/)