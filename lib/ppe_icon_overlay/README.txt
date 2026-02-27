To prepare,
```
apt install -y libopencv-dev pkg-config

apt-get install libglib2.0-dev libglib2.0-0
```

To build,

```
g++ -shared -fPIC ppe_icon_overlay.cpp -o libppe_icon_overlay.so \
    `pkg-config --cflags --libs opencv4 glib-2.0 gstreamer-1.0` \
    -I/opt/nvidia/deepstream/deepstream-7.0/sources/includes \
    -I/usr/local/cuda/include \
    -L/opt/nvidia/deepstream/deepstream-7.0/lib \
    -lnvdsgst_meta \
    -lnvbufsurface \
    -lnvbufsurftransform
```