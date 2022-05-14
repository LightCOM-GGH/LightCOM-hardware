# Repo for the embedded hardware

## SETUP
setup the venv:
```
python3 -m venv env
```

activate venv:
```
env\Scripts\activate.bat
```

install all the requirements:
```
pip install -r requirements.txt
```

YOLO stuff from here:
https://lindevs.com/yolov4-object-detection-using-tensorflow-2/



## Usage
### Demo images from data/ folder
```
cd source
python main.py demo
```

### Run with the camera
```
cd source
python main.py
```