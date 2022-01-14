## Installation

### Requirements

Install requirements:

```bash
pip install -r requirements.txt
```

Other requirements:

- apex
- cocoapi
- GCC >= 4.9

### Step by step installation of modules

```bash

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# compile and install maskrcnn-module
python setup.py build develop

```

