[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/BegzSP5S)

# Setup

1. Clone the repo
2. `cd` into the repo directory
3. Setup and activate a virtual env **(Python 3.12)**
4. `pip install -r requirements.txt`

# Camera-Based Touch Sensor

Documentation on this task can be found in [documentation.md](documentation.md).  

**How to launch the application:**
```bash
# Complete the setup steps above
python touch_input.py -c 0 -d
```

## Command Line Parameters

| Parameter | Short Option | Default | Description |
|-----------|--------------|---------|-------------|
| `--video-id` | `-c` | `0` | ID of the webcam you want to use |
| `--debug` | `-d` | `False` | Enable debug mode (Whether to show preview window or not) |
| `--host` | | `127.0.0.1` | Host to broadcast events to (DIPPID) |
| `--port` | `-p` | `5700` | Port to broadcast events to (DIPPID) |
| `--calibration-frames` | | `5` | Number of frames to use for brightness calibration |