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

* `ego_centric_analysis.py`: The master execution script running the core pipeline loop.
* `config.py`: Centralizes all configurable hyperparameters, thresholds, and data paths.
* `utils.py`: Provides utility functions for driver eye-gaze coordinate parsing, fixation math, projections, and plot layouts.
* `models.py`: Defines the `TemporalObjectTracker` and the NetworkX-based `PersistentKnowledgeGraph` classes.
* `data_samples/`: A placeholder directory for verifying files, containing sample gaze info (`.txt`), scene images (`.png`), scene depth maps (`.npy`), and face images (`.png`).
* `pretrained_models/`: Target workspace for storing pretrained YOLOv8s weights and the SAM2.1 Hiera Large checkpoint.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/declan-ebz/EGH490-2-Gaze-Based-SA-Estimation-Framework.git
cd EGH490-2-Gaze-Based-SA-Estimation-Framework

```

### 2. Google Colab Environment

This project is optimized to run in a Google Colab notebook environment. Ensure you have access to a GPU runtime (`Runtime > Change runtime type > GPU`).

> [!IMPORTANT]
> **Hardware Requirement**: This architecture requires a GPU runtime with a minimum of **15GB VRAM** (e.g., Google Colab T4, L4, or A100 instances) to seamlessly handle the combined memory overhead of YOLOv8, SAM2, and the quantized Qwen2.5-VL pipelines without throwing Out-Of-Memory (OOM) errors.

### 3. Dependencies

Install the required system and Python packages via your terminal or a Colab cell:

```bash
pip install -r requirements.txt

```

Key dependencies track:

* `ultralytics` (for YOLOv8 spatial object detection)
* `sam2` (Segment Anything Model 2 temporal video propagation library)
* `transformers`, `accelerate`, `bitsandbytes` (for 4-bit VLM quantization and pipeline deployment)
* `google-generativeai` (for Gemini API orchestration)
* `opencv-python`, `numpy`, `matplotlib`, `networkx`

### 4. Data and Model Paths

**Data Layout:**
The `data_samples/` directory expects a structured target folder matching your runtime ingestion pathways (e.g., `/Subject26_2_data`). This folder must contain `face_ims`, `gaze_info`, `scene_ims`, and `scene_depth` subdirectories. You can mount your Google Drive with this data asset or upload target samples directly into your environment workspace.

**Model Weights:**

* **YOLOv8s**: Ensure your pre-trained YOLO model is placed at the path configured in your script (`pretrained_models/yolov8s.pt`).
* **SAM2**: The `sam2.1_hiera_large.pt` checkpoint file should be downloaded and placed directly inside `pretrained_models/`. Your execution workspace provides standard download commands for this.

Verify or update all absolute local string indices inside `config.py` to seamlessly align with your system storage before launching.

### 5. Gemini API Key

To enable VLM features, multi-stage instructor evaluation prompts, and automated situational awareness scoring sweeps, you require a valid Google Gemini API Key. Store it securely in Colab's secrets manager side panel under the token key name: `GOOGLE_API_KEY`.

> [!NOTE]
> If executing this pipeline framework outside of a Google Colab environment (such as a local Linux shell or dedicated development server), replace the `userdata.get()` parsing blocks inside `ego_centric_analysis.py` with standard environment parsing logic: `os.environ.get("GOOGLE_API_KEY")`.

---

## Usage

1. **Execute the Modular Pipeline (`ego_centric_analysis.py`)**:
Run the master orchestrator to sequentially spin up and evaluate your architecture layers:
* Sets environment setups and ingests configuration thresholds from `config.py`.
* Ingests and processes gaze data telemetry arrays to detect true fixations.
* Run object localization and tracking layers with active Qwen2.5-VL semantic triage.
* Propagates mask overlays and frame prompts smoothly via SAM2 temporal tracking loops.
* Compiles and records nodes inside the active NetworkX `PersistentKnowledgeGraph`.
* Fires multi-stage Gemini loops to parse driver advice logs and calculate safety metrics.
* Saves final frame visual metrics and assembles video clips.



```bash
python ego_centric_analysis.py

```

2. **Generated Outputs**:
The execution loop produces the following outputs:
* `gaze_prompts.json`: Compiled bounding and point tracking cues formatted for SAM2 consumption.
* `scene_segmentations/`: Workspace directory containing system visualization maps for each frame (`eval_map_*.jpg`).
* `scene_segmentations/gemini_advice.json`: A structural JSON database logging selected driver alerts and safety outputs generated per frame index.
* `scene_segmentations/full_driving_scene_graph.mp4`: A compiled, high-definition video summary mapping out camera view targets, fixation networks, and rolling memory timelines simultaneously.



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