Dependências necessárias no Linux (Google Colab):

1. Atualizar pacotes:
   sudo apt-get update

2. Instalar o OpenCV:
   sudo apt-get install -y libopencv-dev python3-opencv

3. Instalar o compilador C++:
   sudo apt-get install -y build-essential

4. Instalar ferramentas adicionais (opcional, para suporte a vídeos):
   sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev

5. Fazer download de modelos de super-resolução:
   - Exemplo de download do modelo EDSR:
     wget -O EDSR_x4.pb https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/model/EDSR_x4.pb

6. Compilar o código:
   - Utilize o seguinte comando para compilar:
     g++ -o remasterizador remasterizador.cpp `pkg-config --cflags --libs opencv4`

7. Executar o programa:
   - Após compilar, execute o programa passando os argumentos necessários:
     ./remasterizador <input_video> <output_video> <model_path> <model_name> <scale>
   - Exemplo:
     ./remasterizador input.mp4 output.mp4 EDSR_x4.pb edsr 4