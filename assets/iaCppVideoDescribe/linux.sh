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

add_executable(iaCppVideoDescribe iaCppVdSh/src/iaCppVideoDescribe.cpp)

target_link_libraries(iaCppVideoDescribe \${OpenCV_LIBS} \${TORCH_LIBRARIES} c10 c10_cuda)

set(CMAKE_EXE_LINKER_FLAGS "\${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,\${TORCH_INSTALL_PREFIX}/lib")
EOF

export CMAKE_PREFIX_PATH=$LIBTORCH_DIR

# Verificar e corrigir o código-fonte
if grep -q "#include <torch/parallel.h>" "iaCppVdSh/src/iaCppVideoDescribe.cpp"; then
    echo "Corrigindo o código-fonte para remover torch/parallel.h..."
    sed -i '/#include <torch\/parallel.h>/d' "iaCppVdSh/src/iaCppVideoDescribe.cpp"
    echo "#include <ATen/Parallel.h>" >> "iaCppVdSh/src/iaCppVideoDescribe.cpp"
fi

echo "Compilando Torch"
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR ..
make -j$(nproc)

echo "Gerando modelo pré treinado..."

# Diretório onde o modelo será salvo
MODEL_DIR="./"
MODEL_PATH="${MODEL_DIR}/yolov5s.pt"
TORCHSCRIPT_MODEL_PATH="${MODEL_DIR}/yolov5s_scripted.pt"

# Verificar se o diretório de modelos existe, caso contrário, criar
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p "$MODEL_DIR"
fi

# Baixar o modelo YOLOv5 pré-treinado (modelo mais leve)
echo "Baixando o modelo YOLOv5..."
rm -rf $MODEL_PATH  # Remover o modelo existente, se houver
wget -O "$MODEL_PATH" https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt

# Verificar se o PyTorch e o repositório YOLOv5 estão instalados para converter para TorchScript
echo "Verificando dependências para gerar o modelo pre treinado..."

# Instalar Python 3.8
echo "Instalando Python 3.8..."
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv

# Instalar pip para Python 3.8
apt-get install -y python3.8-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py

# Instalar dependências (PyTorch e YOLOv5)
rm -rf yolov5  # Remover o repositório existente, se houver
git clone https://github.com/ultralytics/yolov5  # clone
python3.8 -m pip install -r yolov5/requirements.txt  # install

# Script para converter o modelo para o formato TorchScript
echo "Convertendo o modelo para o formato TorchScript..."

# Python script para carregar o modelo YOLOv5 e exportá-lo como TorchScript
cat <<EOF > "build_model.py"
import torch
from models.common import DetectMultiBackend

# Carregar o modelo YOLOv5
model = DetectMultiBackend('$MODEL_PATH', device='cpu')  # Use 'cuda' se estiver usando GPU

# Converter o modelo para o formato TorchScript
scripted_model = model.model.to('cpu').half().eval()  # Mover para CPU e modo eval
scripted_model = torch.jit.script(scripted_model)  # Convertendo para TorchScript

# Salvar o modelo TorchScript
scripted_model.save('$TORCHSCRIPT_MODEL_PATH')
EOF

python3.8 build_model.py

# Verificação final
if [ -f "$TORCHSCRIPT_MODEL_PATH" ]; then
    echo "Modelo TorchScript salvo com sucesso em $TORCHSCRIPT_MODEL_PATH."
else
    echo "Erro ao converter o modelo para TorchScript."
fi

# Nome do arquivo de saída
OUTPUT="../iaCppVideoDescribe"

# Caminho para o código-fonte
SOURCE="../src/iaCppVideoDescribe.cpp"

# Flags de compilação
CXX=g++
CXXFLAGS="-std=c++17 -fopenmp -O2"
INCLUDE_FLAGS="-I${LIBTORCH_DIR}/include -I${LIBTORCH_DIR}/include/torch/csrc/api/include -I${OpenCV_INCLUDE_DIRS}"
LIB_FLAGS="-L${LIBTORCH_DIR}/lib -ltorch -ltorch_cpu -lc10 -L${OpenCV_LIBS} -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui"

# Checando se os caminhos estão configurados corretamente
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Erro: Diretório da LibTorch não encontrado em $LIBTORCH_DIR"
    exit 1
fi

if [ ! -d "$OpenCV_INCLUDE_DIRS" ]; then
    echo "Erro: Diretório de cabeçalhos do OpenCV não encontrado em $OpenCV_INCLUDE_DIRS"
    exit 1
fi

if [ ! -d "$OPENCV_LIB" ]; then
    echo "Erro: Diretório de bibliotecas do OpenCV não encontrado em $OPENCV_LIB"
    exit 1
fi

# Exportando variáveis para o linker
export LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:$LD_LIBRARY_PATH

# Comando de compilação
echo "Compilando o código..."
$CXX $CXXFLAGS $INCLUDE_FLAGS $SOURCE -o $OUTPUT $LIB_FLAGS

# Verificação do resultado
if [ $? -eq 0 ]; then
    echo "Compilação concluída com sucesso! O executável foi gerado como './$OUTPUT'."
else
    echo "Erro na compilação."
    exit 1
fi

echo "Modelo TorchScript salvo em $TORCHSCRIPT_MODEL_PATH."
echo "Para executar o programa, use: iaCppVdSh/video_pipeline <caminho_do_video> <caminho_do_modelo>"