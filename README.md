# IA CPP Remaster

Projeto de remasterização de videos antigos utilizando C++ e ML no ambiente linux.

O projeto foi feito para o https://colab.research.google.com/

## Ações feitas no video

- Remove ruidos;
- Aumenta resolução;
- Interpola quadros;

## Modelo de ML

### Modelos

#### EDSR

Dados de exemplo: [opencv_super_resolution_EDSR_x4.pb](./assets/opencv_super_resolution_EDSR_x4.pb)

#### FSRCNN

Dados de exemplo: [opencv_super_resolution_FSRCNN_x2.pb](./assets/opencv_super_resolution_FSRCNN_x2.pb)

## Comandos necessários para instalação no Linux

#### Instala biblitoecas
```bash
apt-get update && apt-get -y libopencv-dev python3-opencv build-essential ffmpeg libavcodec-dev libavformat-dev libswscale-dev
```

#### Clona projeto
```bash
git clone https://github.com/SamuelSBJr97/iaCppRemaster.git
```

#### Compila script
```bash
g++ -o iaCppRemaster/iaCppRemaster iaCppRemaster/src/iaCppRemaster.cpp `pkg-config --cflags --libs opencv4`
```

## Utilização

```bash
iaCppRemaster <inputVideoPath> <outputVideoPath> <modelPath> <modelName> <scale> <arquitetura> <targetFPS>
```

1. `inputVideoPath`: O caminho para o vídeo de entrada que você deseja processar.
2. `outputVideoPath`: O caminho para o vídeo de saída onde o vídeo processado será salvo.
3. `modelPath`: O caminho para o arquivo do modelo de super-resolução que você deseja usar.
4. `modelName`: O nome do modelo de super-resolução. Pode ser **EDSR**, **FSRCNN**.
5. `scale`: A escala de super-resolução. Por exemplo, 2 significa que a resolução será aumentada em 2 vezes.
6. `arquitetura`: Indica se será utilizado **CPU** ou **GPU** para o processamento. 
   - Use **gpu** para usar a GPU.
   - Use **cpu** para usar a CPU.
7. `targetFPS`: O FPS alvo para o vídeo de saída.

## Remasterizando um video de 8 segundos

#### Video de exemplo de 8 segundos

Trata-se do filme Nosferatu de 1922.

[input.mkv](./assets/input.mkv)

#### Exemplo de uso no Google Colab

[iaCppRemaster.ipynb](./assets/iaCppRemaster.ipynb)

##### 2K de resolução na CPU com 30FPS no modelo fsrcnn
```bash
iaCppRemaster/iaCppRemaster iaCppRemaster/assets/input.mkv iaCppRemaster/assets/output.mp4 iaCppRemaster/assets/opencv_super_resolution_FSRCNN_x2.pb fsrcnn 2 cpu 30
```

##### 4K de resolução na GPU com 60FPS no modelo edsr
```bash
iaCppRemaster/iaCppRemaster iaCppRemaster/assets/input.mkv iaCppRemaster/assets/output.mp4 iaCppRemaster/assets/opencv_super_resolution_EDSR_x4.pb edsr 2 gpu 60
```