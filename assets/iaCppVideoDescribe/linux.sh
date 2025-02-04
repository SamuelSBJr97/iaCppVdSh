#!/bin/bash

# Script para instalar dependências e compilar o pipeline para Linux com suporte a GPU/CPU

set -e  # Para o script em caso de erro

# Atualizar o sistema
apt update && sudo apt upgrade -y

# Instalar dependências essenciais
apt install -y build-essential cmake git wget unzip libopencv-dev

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

cat <<EOF > "CMakeLists.txt"
# Definição da versão mínima do CMake
cmake_minimum_required(VERSION 3.10)

# Nome do projeto
project(iaCppVdSh/src/iaCppVideoDescribe)

# Configuração do C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Caminhos para LibTorch
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")

# Caminhos para OpenCV
find_package(OpenCV REQUIRED)

# Adicionar o arquivo-fonte
add_executable(iaCppVideoDescribe iaCppVdSh/src/iaCppVideoDescribe.cpp)

# Incluir diretórios
include_directories(/usr/include/opencv4)
find_package(Torch REQUIRED)

# Vincular bibliotecas
target_link_libraries(iaCppVideoDescribe "${LIBTORCH_DIR}/libtorch_cuda.so ${LIBTORCH_DIR}/libtorch_cpu.so ${LIBTORCH_DIR}/libc10.so")

# Exportar variáveis de ambiente para o runtime
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath,${LIBTORCH_DIR}/libtorch_cuda.so ${LIBTORCH_DIR}/libtorch_cpu.so ${LIBTORCH_DIR}/libc10.so")
EOF

# Configurar variáveis de ambiente para Libtorch
export CMAKE_PREFIX_PATH=$LIBTORCH_DIR
# Compilar o código
echo "Compilando o código..."
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR ..
make

# Mensagem de sucesso
echo "CMAKE Libtorch"

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
wget -O "$MODEL_PATH" https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt

# Verificar se o PyTorch e o repositório YOLOv5 estão instalados para converter para TorchScript
echo "Verificando dependências..."

# Instalar dependências (PyTorch e YOLOv5)
pip install torch
pip install git+https://github.com/ultralytics/yolov5.git

# Script para converter o modelo para o formato TorchScript
echo "Convertendo o modelo para o formato TorchScript..."

# Python script para carregar o modelo YOLOv5 e exportá-lo como TorchScript
python3 - <<EOF
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

# Caminho para o diretório da LibTorch
LIBTORCH_DIR="/path/to/libtorch"

# Caminho para o OpenCV (caso necessário, ajuste para o sistema)
OPENCV_INCLUDE="/usr/include/opencv4"
OPENCV_LIB="/usr/lib"

# Flags de compilação
CXX=g++
CXXFLAGS="-std=c++17 -fopenmp -O2"
INCLUDE_FLAGS="-I${LIBTORCH_DIR}/include -I${LIBTORCH_DIR}/include/torch/csrc/api/include -I${OPENCV_INCLUDE}"
LIB_FLAGS="-L${LIBTORCH_DIR}/lib -ltorch -ltorch_cpu -lc10 -L${OPENCV_LIB} -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui"

# Checando se os caminhos estão configurados corretamente
if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Erro: Diretório da LibTorch não encontrado em $LIBTORCH_DIR"
    exit 1
fi

if [ ! -d "$OPENCV_INCLUDE" ]; then
    echo "Erro: Diretório de cabeçalhos do OpenCV não encontrado em $OPENCV_INCLUDE"
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