#!/bin/bash
apt-get update
apt-get install -y texlive-latex-extra imagemagick ffmpeg ghostscript
apt-get clean
rm -rf /var/lib/apt/lists/*
sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cudatoolkit=11.8 -c pytorch -c nvidia
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117