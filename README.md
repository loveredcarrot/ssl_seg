# ssl_seg
Semi-supervised semantic segmentation for medical imaging
this project is for image semantic segmentation (just 2 classes)
image: RGB label:mask

## run the project
if you use anaconda
conda create -n your_env_name python=3.6
- pip install torch==0.4.1
- pip install torchvision==0.2.0 (numpy-1.19.5 pillow-8.1.2 six-1.15.0 torchvision-0.2.0)
- pip install numpy==1.19.0 (useless)
- pip install medpy==0.4.0 (SimpleITK-2.0.2 medpy-0.4.0 scipy-1.5.4)
- pip install tensorboardX (protobuf-3.15.6 tensorboardX-2.1)
- pip install tqdm (tqdm-4.59.0)
- pip install opencv-python
- pip install matplotlib (cycler-0.10.0 kiwisolver-1.3.1 matplotlib-3.3.4 pyparsing-2.4.7 python-dateutil-2.8.1)
- pip install pillow

## some semi-supervised method implement
mean teacher
ICT
UA-MT


