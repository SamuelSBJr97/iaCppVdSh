#!/bin/bash

# Atualizar e instalar dependências básicas
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev

# Instalar o OpenCV diretamente dos repositórios do Ubuntu
sudo apt install -y libopencv-dev python3-opencv

g++ -o iaCppRemaster/iaCppRemaster iaCppRemaster/iaCppRemaster.cpp `pkg-config --cflags --libs opencv4`