ok#!/bin/bash

# Atualizar e instalar dependências básicas
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev

# Instalar o OpenCV diretamente dos repositórios do Ubuntu
sudo apt install -y libopencv-dev python3-opencv libomp-dev

echo "Baixando e instalando OpenVINO..."
OPENVINO_VERSION="2024.0.0.16972"
OPENVINO_FILE="l_openvino_toolkit_${OPENVINO_VERSION}_x86_64.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/linux/$OPENVINO_FILE"

echo "Configurando OpenVINO..."
echo "source /opt/intel/openvino_2024/setupvars.sh" >> ~/.bashrc
source /opt/intel/openvino_2024/setupvars.sh

echo "Limpeza de arquivos temporários..."
rm -rf "$OPENVINO_FILE" "l_openvino_toolkit_${OPENVINO_VERSION}_x86_64"

g++ -o /content/iaCppRemaster/iaCppRemaster /content/iaCppRemaster/src/iaCppRemaster.cpp `pkg-config --cflags --libs opencv4`

g++ -o /content/iaCppRemaster/iaCppVerticalFill /content/iaCppRemaster/src/iaCppVerticalFill.cpp std=c++17 `pkg-config --cflags --libs opencv4` -fopenmp -I/opt/intel/openvino_2024/runtime/include -L/opt/intel/openvino_2024/runtime/lib/intel64 -lopenvino