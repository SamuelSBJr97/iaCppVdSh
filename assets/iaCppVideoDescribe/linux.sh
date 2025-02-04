#!/bin/bash

# Script para instalar dependências e compilar o pipeline para Linux com suporte a GPU/CPU

set -e  # Para o script em caso de erro

mkdir -p iaCppVdSh/build && cd iaCppVdSh/build

# Atualizar o sistema
apt update && apt upgrade -y

# Instalar dependências essenciais
apt install -y build-essential cmake git wget unzip libopencv-dev

# Instalar OpenCV
if ! pkg-config --exists opencv4; then
    echo "Instalando OpenCV..."
    apt install -y libopencv-dev
else
    echo "OpenCV já está instalado."
fi

# Instalar Libtorch (PyTorch para C++)
LIBTORCH_DIR="/usr/local/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Instalando Libtorch..."
    wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip -O libtorch.zip
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

find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA encontrado: ${CUDA_VERSION}")
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARIES})
else()
    message(FATAL_ERROR "CUDA não foi encontrado! Certifique-se de que ele está instalado corretamente.")
endif()

include_directories(
    \${OpenCV_INCLUDE_DIRS}
    \${TORCH_INCLUDE_DIRS}
    ${LIBTORCH_DIR}/include
    ${LIBTORCH_DIR}/include/torch/csrc/api/include
    ${LIBTORCH_DIR}/include/torch
    ${LIBTORCH_DIR}/include/ATen
)

add_executable(iaCppVideoDescribe ../../iaCppVdSh/src/iaCppVideoDescribe.cpp)

target_link_libraries(iaCppVideoDescribe \${OpenCV_LIBS} \${TORCH_LIBRARIES} c10 c10_cuda)

set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,\${TORCH_INSTALL_PREFIX}/lib")
EOF

export CMAKE_PREFIX_PATH=$LIBTORCH_DIR

# Verificar e corrigir o código-fonte
if grep -q "#include <torch/parallel.h>" "../../iaCppVdSh/src/iaCppVideoDescribe.cpp"; then
    echo "Corrigindo o código-fonte para remover torch/parallel.h..."
    sed -i '/#include <torch\/parallel.h>/d' "../../iaCppVdSh/src/iaCppVideoDescribe.cpp"
    echo "#include <ATen/Parallel.h>" >> "../../iaCppVdSh/src/iaCppVideoDescribe.cpp"
fi

echo "Compilando iaCppVideoDescribe"
rm -rf build
mkdir -p build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR -B build -S .
make -C build -j$(nproc)

mv build/iaCppVideoDescribe ../

echo "Baixando modelo pré treinado..."

# Diretório onde o modelo será salvo
MODEL_PATH="../assets/iaCppVideoDescribe/yolov5s.pt"

# Baixar o modelo YOLOv5 pré-treinado (modelo mais leve)
echo "Baixando o modelo yolov5s.pt..."
rm -rf $MODEL_PATH  # Remover o modelo existente, se houver
wget -O "$MODEL_PATH" https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt

# Verificação final
if [ -f "$MODEL_PATH" ]; then
    echo "Modelo TorchScript salvo com sucesso em $MODEL_PATH."
else
    echo "Erro ao converter o modelo para TorchScript."
fi

# Verificar se o PyTorch e o repositório YOLOv5 estão instalados para converter para TorchScript
echo "Verificando dependências para gerar o modelo pre treinado..."

# Instalar Python 3.8
echo "Instalando Python 3.8..."
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.8 python3.8-venv

# Instalar pip para Python 3.8
echo "Instalando pip para Python 3.8..."
apt-get install -y python3.8-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py

# Instalar dependências (PyTorch e YOLOv5)
echo "Instalando https://github.com/ultralytics/yolov5..."
rm -rf yolov5  # Remover o repositório existente, se houver
git clone https://github.com/ultralytics/yolov5  # clone
python3.8 -m pip install -r yolov5/requirements.txt  # install

# Script para converter o modelo para o formato TorchScript
echo "Convertendo o modelo para o formato TorchScript..."

# Python script para carregar o modelo YOLOv5 e exportá-lo como TorchScript
cat <<EOF > "convert_model.py"
import sys
sys.path.append('yolov5')  # Adicionar o repositório YOLOv5 ao PYTHONPATH
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Caminho para o modelo YOLOv5
model_path = '${MODEL_PATH}'

# Carregar o modelo YOLOv5
device = select_device('cpu')
model = DetectMultiBackend(model_path, device=device)  # Use 'cuda' se estiver usando GPU

# Converter o modelo para o formato TorchScript
scripted_model = torch.jit.script(model.model)

# Salvar o modelo TorchScript
scripted_model.save('${TORCHSCRIPT_MODEL_PATH}')
print("Modelo convertido e salvo como '${TORCHSCRIPT_MODEL_PATH}'")
EOF

# Executar o script Python no diretório correto
mv convert_model.py yolov5/
python3.8 yolov5/convert_model.py

# Verificação final
if [ -f "$TORCHSCRIPT_MODEL_PATH" ]; then
    echo "Modelo TorchScript salvo com sucesso em $TORCHSCRIPT_MODEL_PATH."
else
    echo "Erro ao converter o modelo para TorchScript."
fi

echo "Modelo TorchScript salvo em $MODEL_PATH."
echo "Para executar o programa, use: iaCppVdSh/video_pipeline <caminho_do_video> <caminho_do_modelo>"