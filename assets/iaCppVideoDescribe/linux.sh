#!/bin/bash

# Script para instalar dependências e compilar o pipeline para Linux com suporte a GPU/CPU

set -e  # Para o script em caso de erro

LOG_FILE="install_logs.txt"

# Função para logar mensagens
log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Função para logar erros
log_error() {
    echo "$1" | tee -a "$LOG_FILE" >&2
}

mkdir -p iaCppVdSh/build && cd iaCppVdSh/build

# Atualizar o sistema
log "Atualizando o sistema..."
apt update | tee -a "$LOG_FILE"
sudo apt upgrade -y | tee -a "$LOG_FILE"

# Instalar dependências essenciais
log "Instalando dependências essenciais..."
apt install -y build-essential cmake git wget unzip libopencv-dev | tee -a "$LOG_FILE"

# Instalar OpenCV
if ! pkg-config --exists opencv4; then
    log "Instalando OpenCV..."
    sudo apt install -y libopencv-dev | tee -a "$LOG_FILE"
else
    log "OpenCV já está instalado."
fi

# Instalar Libtorch (PyTorch para C++)
LIBTORCH_DIR="/usr/local/libtorch"
if [ ! -d "$LIBTORCH_DIR" ]; then
    log "Instalando Libtorch..."
    wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip -O libtorch.zip | tee -a "$LOG_FILE"
    unzip libtorch.zip -d /usr/local/ | tee -a "$LOG_FILE"
    rm libtorch.zip | tee -a "$LOG_FILE"
else
    log "Libtorch já está instalado."
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
    message(FATAL_ERROR "LibTorch não encontrada.")
endif()

add_executable(iaCppVideoDescribe ../src/iaCppVideoDescribe.cpp)
target_link_libraries(iaCppVideoDescribe "\${TORCH_LIBRARIES}")
EOF

log "Compilando Torch"
rm -rf build && mkdir -p build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_DIR -B build -S . | tee -a "$LOG_FILE"
make -C build -j$(nproc) | tee -a "$LOG_FILE"

log "Gerando modelo pré treinado..."

# Diretório onde o modelo será salvo
MODEL_DIR="./"
MODEL_PATH="${MODEL_DIR}/yolov5s.pt"
TORCHSCRIPT_MODEL_PATH="${MODEL_DIR}/yolov5s_scripted.pt"

# Verificar se o diretório de modelos existe, caso contrário, criar
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p "$MODEL_DIR"
fi

# Baixar o modelo YOLOv5 pré-treinado (modelo mais leve)
log "Baixando o modelo YOLOv5..."
rm -rf $MODEL_PATH  # Remover o modelo existente, se houver
wget -O "$MODEL_PATH" https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt | tee -a "$LOG_FILE"

# Verificar se o PyTorch e o repositório YOLOv5 estão instalados para converter para TorchScript
log "Verificando dependências para gerar o modelo pre treinado..."

# Instalar Python 3.8
log "Instalando Python 3.8..."
sudo apt-get update | tee -a "$LOG_FILE"
sudo apt-get install -y software-properties-common | tee -a "$LOG_FILE"
sudo add-apt-repository -y ppa:deadsnakes/ppa | tee -a "$LOG_FILE"
sudo apt-get update | tee -a "$LOG_FILE"
sudo apt-get install -y python3.8 python3.8-venv | tee -a "$LOG_FILE"

# Instalar pip para Python 3.8
log "Instalando pip para Python 3.8..."
sudo apt-get install -y python3.8-distutils | tee -a "$LOG_FILE"
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py | tee -a "$LOG_FILE"
sudo python3.8 get-pip.py | tee -a "$LOG_FILE"

# Instalar dependências (PyTorch e YOLOv5)
log "Instalando https://github.com/ultralytics/yolov5..."
rm -rf yolov5  # Remover o repositório existente, se houver
git clone https://github.com/ultralytics/yolov5 | tee -a "$LOG_FILE"  # clone
python3.8 -m pip install -r yolov5/requirements.txt | tee -a "$LOG_FILE"  # install

# Script para converter o modelo para o formato TorchScript
log "Convertendo o modelo para o formato TorchScript..."

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

python3.8 build_model.py | tee -a "$LOG_FILE"

# Verificação final
if [ -f "$TORCHSCRIPT_MODEL_PATH" ]; then
    log "Modelo TorchScript salvo com sucesso em $TORCHSCRIPT_MODEL_PATH."
else
    log_error "Erro ao converter o modelo para TorchScript."
fi

# Caminho para o código-fonte
SOURCE="../src/iaCppVideoDescribe.cpp"

# Flags de compilação
CXX=g++
CXXFLAGS="-std=c++17 -fopenmp -O2"
INCLUDE_FLAGS="-I${LIBTORCH_DIR}/include -I${LIBTORCH_DIR}/include/torch/csrc/api/include -I${OpenCV_INCLUDE_DIRS}"
LIB_FLAGS="-L${LIBTORCH_DIR}/lib -ltorch -ltorch_cpu -lc10 -L${OpenCV_LIBS} -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui"

# Checando se os caminhos estão configurados corretamente
if [ ! -d "$LIBTORCH_DIR" ]; then
    log_error "Erro: Diretório da LibTorch não encontrado em $LIBTORCH_DIR"
    exit 1
fi

if [ ! -d "$OpenCV_INCLUDE_DIRS" ]; then
    log_error "Erro: Diretório de cabeçalhos do OpenCV não encontrado em $OpenCV_INCLUDE_DIRS"
    exit 1
fi

if [ ! -d "$OPENCV_LIB" ]; then
    log_error "Erro: Diretório de bibliotecas do OpenCV não encontrado em $OPENCV_LIB"
    exit 1
fi

# Exportando variáveis para o linker
export LD_LIBRARY_PATH=${LIBTORCH_DIR}/lib:$LD_LIBRARY_PATH

# Comando de compilação
log "Compilando o código..."
$CXX $CXXFLAGS $INCLUDE_FLAGS $SOURCE -o iaCppVideoDescribe $LIB_FLAGS | tee -a "$LOG_FILE"

# Verificação do resultado
if [ $? -eq 0 ]; then
    log "Compilação concluída com sucesso! O executável foi gerado como './iaCppVideoDescribe'."
else
    log_error "Erro na compilação."
    exit 1
fi

log "Modelo TorchScript salvo em $TORCHSCRIPT_MODEL_PATH."
log "Para executar o programa, use: iaCppVdSh/video_pipeline <caminho_do_video> <caminho_do_modelo>"