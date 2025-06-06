# Real-Time Person Tracking with YOLOv8 and DeepSORT

This project implements a real-time person tracking system using the YOLOv8 model for object detection and the DeepSORT algorithm for tracking. It processes a video file to detect and assign a unique ID to each person, tracking them across frames.

## Prerequisites

-   **Python 3.9.0**: This project is built and tested with Python 3.9.0.
    -   You can download the specific version from the [official Python website](https://www.python.org/downloads/release/python-390/).
    -   To check your Python version, run the following command in your terminal:
        ```bash
        python --version
        ```

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Clone the DeepSORT Repository

This project requires the official DeepSORT library. You must clone it **inside** the project directory you just created.

```bash
git clone https://github.com/nwojke/deep_sort.git
```

After this step, your project directory should look like this:

```
your-repo-name/
├── deep_sort/      <-- The repository you just cloned
├── model_data/
├── main.py
├── tracker.py
└── ... other files
```

**3. Create and Activate a Virtual Environment**

It's highly recommended to use a virtual environment to manage project dependencies.

-   Create the environment (we'll name it `venv`):
    ```bash
    py -3.9 -m venv venv39 
    or
    python -m venv venv
    ```

-   Activate the environment:
    -   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

**4. Install Required Packages**

A `requirements.txt` file is provided to install the correct versions of all necessary packages.

```bash
pip install -r requirements.txt
```

## Project Configuration

Before running the script, you need to set up the model and data folders, as they are not included in the repository.

**1. YOLOv8 Model**

The `yolov8n.pt` model file is required for detection. The script is designed to **download this file automatically** the first time you run it. No manual action is needed.

**2. DeepSORT Model**

The DeepSORT Re-ID model (`mars-small128.pb` and related files) should be located in the `model_data` folder, which is included in this repository.

**3. Input Video**

The script expects an input video file.

-   Create a folder named `data` in the root of the project directory.
-   Place your input video file inside this `data` folder. The script is currently configured to look for a file named `people.mp4`.
    ```
    your-repo-name/
    ├── data/
    │   └── people.mp4   <-- Place your video here
    ├── ... (other files)
    ```

## Running the Tracker

Once the environment is set up and the input video is in place, run the main script from the root directory of the project:

```bash
python main.py
```

-   A window titled "Video Tracking" should appear, showing the video with bounding boxes and tracking IDs drawn on detected persons.
-   Press the **'q'** key on your keyboard while the video window is active to stop the script.

## Output

The processed video, with tracking visualizations, will be saved as `out.mp4` in the root directory of the project.