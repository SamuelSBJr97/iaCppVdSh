#!/bin/bash

# Script para instalar dependências e compilar o pipeline para Linux com suporte a GPU/CPU

set -e  # Para o script em caso de erro

# Atualizar o sistema
apt update && sudo apt upgrade -y

# Instalar dependências essenciais
apt install -y build-essential cmake git wget unzip libopencv-dev

# Instalar OpenCV
if ! pkg-config --exists opencv4; then
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

cat <<EOF > "CMakeLists.txt"
cmake_minimum_required(VERSION 3.10)

project(iaCppVideoDescribe)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Configuração do projeto iaCppVideoDescribe...")

set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED)
if (Torch_FOUND)
    message(STATUS "LibTorch encontrada em: \${Torch_DIR}")
else()
    message(FATAL_ERROR "LibTorch não foi encontrada! Verifique se o caminho está correto.")
endif()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV encontrado em: \${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV não foi encontrado! Certifique-se de que ele está instalado.")
endif()

include_directories(
    \${OpenCV_INCLUDE_DIRS}
    \${TORCH_INCLUDE_DIRS}
    ${LIBTORCH_DIR}/include
    ${LIBTORCH_DIR}/include/torch/csrc/api/include
    ${LIBTORCH_DIR}/include/torch
)

add_executable(iaCppVideoDescribe iaCppVdSh/src/iaCppVideoDescribe.cpp)

target_link_libraries(iaCppVideoDescribe \${OpenCV_LIBS} \${TORCH_LIBRARIES} c10)

set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,\${TORCH_INSTALL_PREFIX}/lib")
EOF

export CMAKE_PREFIX_PATH=$LIBTORCH_DIR

echo "Compilando o código..."
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR ..
make -j$(nproc)

echo "Compilação concluída com sucesso!"

echo "Modelo TorchScript salvo em $TORCHSCRIPT_MODEL_PATH."
echo "Para executar o programa, use: iaCppVdSh/video_pipeline <caminho_do_video> <caminho_do_modelo>"