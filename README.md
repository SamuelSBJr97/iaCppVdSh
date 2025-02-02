# IA CPP Remaster

Projeto de remasterização de videos antigos utilizando C++ no ambiente linux.

O projeto foi feito para o https://colab.research.google.com/

## Ações feitas

- Remove ruidos;
- Interpolação de quadros;
- Aumenta resolução;

## Modelo de treinamento para aumentar resolução

[opencv_super_resolution_EDSR_x4.pb](https://github.com/SamuelSBJr97/iaCppRemaster/blob/main/assets/opencv_super_resolution_EDSR_x4.pb)

## Video de exemplo de 8 segundos

Trata-se do filme Nosferatu de 1922.

[input.mkv](https://github.com/SamuelSBJr97/iaCppRemaster/blob/main/assets/input.mkv)

## Exemplo de uso no Google Colab

[iaCppRemaster.ipynb](https://colab.research.google.com/github/SamuelSBJr97/iaCppRemaster/blob/main/iaCppRemaster.ipynb)

## Comandos necessários

### Instala biblitoecas
```bash
apt-get update && apt-get -y libopencv-dev python3-opencv build-essential ffmpeg libavcodec-dev libavformat-dev libswscale-dev
```

### Clona projeto
```bash
git clone https://github.com/SamuelSBJr97/iaCppRemaster.git
```

### Compila script
```bash
g++ -o iaCppRemaster/iaCppRemaster iaCppRemaster/src/iaCppRemaster.cpp `pkg-config --cflags --libs opencv4`
```

### Testa o script gerado remasterizando um video de 8 segundos

#### 4K de resolução
```bash
iaCppRemaster/iaCppRemaster iaCppRemaster/assets/input.mkv iaCppRemaster/assets/output.mp4 iaCppRemaster/assets/opencv_super_resolution_EDSR_x4.pb edsr 4
```

#### 2K de resolução
```bash
iaCppRemaster/iaCppRemaster iaCppRemaster/assets/input.mkv iaCppRemaster/assets/output.mp4 iaCppRemaster/assets/opencv_super_resolution_FSRCNN_x2.pb fsrcnn 2
```