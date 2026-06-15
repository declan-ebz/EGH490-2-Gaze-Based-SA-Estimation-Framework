# Ego-Centric Driving Scene Understanding and Situational Awareness Framework

## Overview
This repository presents a novel framework for real-time ego-centric driving scene understanding, focusing on a driver's situational awareness. It integrates computer vision models (YOLOv8 for object detection, SAM2 for video object propagation) with visual language models (VLMs) and a persistent knowledge graph to analyze driver gaze, contextualize objects, and assess cognitive state. The system aims to simulate and evaluate a driver's attention and awareness in complex driving scenarios.

## Key Features
-   **Gaze Fixation Detection**: Processes raw gaze data to identify and localize driver fixations within the scene.
-   **Dynamic Object Tracking**: Utilizes YOLOv8 for object detection and a custom `TemporalObjectTracker` for maintaining object identities across video frames.
-   **Semantic Scene Interpretation (VLM Integration)**: Employs a Visual Language Model (Qwen2.5-VL-7B-Instruct) to provide rich, human-like descriptions and relationships (EGO-centric and object-to-object) for critical scene elements, classifying them into 'imminent', 'latent', or 'point of interest' hazards.
-   **SAM2-based Video Object Propagation**: Leverages SAM2 (Segment Anything Model 2) to propagate initial object detections and gaze prompts through video sequences, generating high-quality masks and temporal coherence.
-   **Persistent Knowledge Graph**: A `PersistentKnowledgeGraph` maintains a dynamic mental model of the environment, tracking object states (active, remembered, ghosted) and VLM-derived relationships over time.
-   **Situational Awareness Scoring**: A Gemini-powered evaluator assesses the driver's situational awareness based on object states, VLM insights, and gaze patterns.
-   **Ego-Centric Visualization**: Generates rich visual summaries including camera view with object highlights, gaze overlay, a dynamic fixation mental map, and a rolling Gantt chart of object lifecycles and cognitive states.
## Repository Structure

-   `ego_centric_analysis.ipynb`: The main Colab notebook demonstrating the full framework.
-   `config.py`: Contains all configurable parameters, data paths, and model configurations.
-   `utils.py`: Provides utility functions for gaze data parsing, fixation detection, mapping, and VRAM management.
-   `models.py`: Defines the `TemporalObjectTracker` and `PersistentKnowledgeGraph` classes.
-   `data_samples/`: A directory containing sample gaze info (`.txt`), scene images (`.png`), scene depth maps (`.npy`), and face images (`.png`).
-   `pretrained_models/`: Stores the pretrained YOLOv8s model weights and the SAM2.1 Hiera Large checkpoint.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/declan-ebz/EGH490-2-Gaze-Based-SA-Estimation-Framework.git
```

### 2. Google Colab Environment
This notebook is designed to run in Google Colab. Ensure you have access to a GPU runtime (`Runtime > Change runtime type > GPU`).

### 3. Dependencies
The notebook automatically installs required Python packages. Key dependencies include:
-   `ultralytics` (for YOLOv8)
-   `sam2` (the Segment Anything Model 2 library)
-   `transformers`, `accelerate`, `bitsandbytes` (for VLM integration)
-   `google-generativeai` (for Gemini API)
-   `opencv-python`, `numpy`, `matplotlib`, `networkx`

### 4. Data and Model Paths

**Data:**
-   The `data_samples/` directory is expected to contain a similar structure to `/content/drive/MyDrive/trainv3/Subject26_2_data` shown in the notebook. This includes `face_ims`, `gaze_info`, `scene_ims`, and `scene_depth` subdirectories. You can either mount your Google Drive with this data or upload samples directly.

**Models:**
-   **YOLOv8s**: A pretrained YOLOv8s model is included in `pretrained_models/yolov8s.pt`.
-   **SAM2**: The `sam2.1_hiera_large.pt` checkpoint should be downloaded and placed in `pretrained_models/`. The notebook provides a `wget` command for this.

Update the paths in `config.py` (and the notebook's initial cells if running interactively) to point to your data and model locations.

### 5. Gemini API Key
To enable VLM features and situational awareness scoring, you'll need a Google Gemini API key. Store it securely in Colab's secrets manager under the name `GOOGLE_API_KEY`.

## Usage

1.  **Run the Colab Notebook (`ego_centric_analysis.ipynb`)**: Execute cells sequentially to:
    -   Install dependencies.
    -   Mount Google Drive and verify data structure.
    -   Process gaze data to detect fixations.
    -   Initialize and run object tracking with VLM integration.
    -   Propagate object and gaze prompts with SAM2.
    -   Construct and update the `PersistentKnowledgeGraph`.
    -   Generate visual analyses, including the ego-centric view, mental map, and Gantt chart.
    -   Obtain Gemini-generated driving advice and situational awareness scores.

2.  **Output**: The notebook generates:
    -   `gaze_prompts.json`: Detected gaze prompts for SAM2.
    -   `scene_segmentations/`: Directory containing visualization outputs for each frame (`eval_map_*.jpg`).
    -   `scene_segmentations/gemini_advice.json`: A JSON file logging all Gemini advice generated per frame.
    -   `scene_segmentations/full_driving_scene_graph.mp4`: A compiled video summary of the analysis.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgements
-   [Ultralytics YOLOv8](https://docs.ultralytics.com/)
-   [Pre-trained YOLOV8 Model](https://github.com/Mkoek213/road_detection_model.git)
-   [Eye-Tracking Data](https://github.com/Kasai2020/look_both_ways)
-   [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2)
-   [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
-   [Google Gemini API](https://ai.google.dev/)
