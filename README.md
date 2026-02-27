# Core V3

## Installation

```
podman compose up -d

podman exec -it deepstream-7-dev-core-v3 /bin/bash

source /venv/bin/activate

pip install -r requirements.txt

cd lib/pyds
pip install pyds-1.1.11-py3-none-linux_x86_64.whl
cd ../..

source /opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

## How to Run

Main Program

```
podman compose start
# or
podman compose up -d

# wait until all container ready

podman exec -it deepstream-7-dev-core-v3 /bin/bash

source /venv/bin/activate

python3 -m core_v3.cli run --storage-path /data --rtmp-server rtmp://rtmp.petrokimia-gresik.com:1935/live --rtmp-publish-query-strings user=petrokimia\&pass=OG6sM2Lsm6RW
```

RTMP Server

```
cd ~/projects/nedo-vision/rtmp

podman compose start mediamtx # ffmpeg-cctv-1 ffmpeg-cctv-x
```

Export RFDETR

```

```
