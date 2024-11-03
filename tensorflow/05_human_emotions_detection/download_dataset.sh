#!/bin/bash

pip install -q kaggle
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
rm kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download muhammadhananasghar/human-emotions-datasethes
unzip human-emotions-datasethes.zip -d datasets