#!/bin/bash

# Script para instalar dependências e compilar o pipeline para Linux com suporte a GPU/CPU

set -e  # Para o script em caso de erro

# Atualizar o sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências essenciais
sudo apt install -y build-essential cmake git wget unzip libopencv-dev

# Instalar OpenCV
if ! pkg-config --modversion opencv4 > /dev/null 2>&1; then
    echo "Instalando OpenCV..."
    sudo apt install -y libopencv-dev
else
    echo "OpenCV já está instalado."
fi

# Instalar Libtorch (PyTorch para C++)
LIBTORCH_DIR="/usr/local/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Instalando Libtorch..."
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip -O libtorch.zip
    unzip libtorch.zip -d /usr/local/
    rm libtorch.zip
else
    echo "Libtorch já está instalado."
fi

# Configurar variáveis de ambiente para Libtorch
export CMAKE_PREFIX_PATH=$LIBTORCH_DIR

# Compilar o código
echo "Compilando o código..."
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR ..
make

# Mensagem de sucesso
echo "Pipeline compilado com sucesso!"
echo "Para executar o programa, use: ./video_pipeline <caminho_do_video> <caminho_do_modelo>"