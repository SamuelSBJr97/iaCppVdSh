#!/bin/bash

# Atualizar e instalar dependências básicas
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev

# Instalar o OpenCV diretamente dos repositórios do Ubuntu
sudo apt install -y libopencv-dev python3-opencv libomp-dev

# Instalando dependências do OpenVINO
echo "Instalando dependências..."
sudo apt install -y lsb-release wget sudo cmake build-essential
sudo apt install -y python3 python3-pip python3-dev
sudo apt install -y libopencv-dev libprotobuf-dev protobuf-compiler
sudo apt install -y libssl-dev
sudo apt install -y libcurl4-openssl-dev
sudo apt install -y git

# Baixando o OpenVINO
echo "Baixando o OpenVINO..."
OPENVINO_VERSION="ubuntu20_2024.0.0.14509.34caeefd078"
OPENVINO_FILE="l_openvino_toolkit_${OPENVINO_VERSION}_x86_64.tgz"
OPENVINO_URL="https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/linux/$OPENVINO_FILE"

wget -O "$OPENVINO_FILE" "$OPENVINO_URL"

# Extraindo o arquivo
tar -xvzf "$OPENVINO_FILE"

# Instalando OpenVINO
cd "l_openvino_toolkit_${OPENVINO_VERSION}_x86_64"
sudo ./install.sh

# Configurando variáveis de ambiente do OpenVINO
echo "Configurando OpenVINO..."
echo "source /opt/intel/openvino_2024/setupvars.sh" >> ~/.bashrc
source /opt/intel/openvino_2024/setupvars.sh

# Limpando arquivos temporários
cd ..
rm -rf "$OPENVINO_FILE" "l_openvino_toolkit_${OPENVINO_VERSION}_x86_64"

echo "Instalação do OpenVINO concluída com sucesso!"

g++ -o /content/iaCppRemaster/iaCppRemaster /content/iaCppRemaster/src/iaCppRemaster.cpp `pkg-config --cflags --libs opencv4`

g++ -o /content/iaCppRemaster/iaCppVerticalFill /content/iaCppRemaster/src/iaCppVerticalFill.cpp std=c++17 `pkg-config --cflags --libs opencv4` -fopenmp -I/opt/intel/openvino_2024/runtime/include -L/opt/intel/openvino_2024/runtime/lib/intel64 -lopenvino