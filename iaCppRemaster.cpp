#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>

using namespace std;
using namespace cv;
using namespace dnn_superres;

void enhanceFrame(const Mat& inputFrame, Mat& outputFrame, const string& modelPath, const string& modelName, int scale) {
    // Inicializar o modelo de super-resolução
    DnnSuperResImpl sr;
    sr.readModel(modelPath);
    sr.setModel(modelName, scale);

    // Aplicar super-resolução ao quadro
    sr.upsample(inputFrame, outputFrame);
}

void processVideo(const string& inputVideoPath, const string& outputVideoPath, const string& modelPath, const string& modelName, int scale) {
    VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        cerr << "Erro ao abrir o vídeo: " << inputVideoPath << endl;
        return;
    }

    // Obter parâmetros do vídeo original
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));

    // Configurar o vídeo de saída
    VideoWriter writer(outputVideoPath, fourcc, fps, Size(frameWidth * scale, frameHeight * scale));
    if (!writer.isOpened()) {
        cerr << "Erro ao criar o vídeo de saída: " << outputVideoPath << endl;
        return;
    }

    Mat frame, enhancedFrame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        enhanceFrame(frame, enhancedFrame, modelPath, modelName, scale);
        writer.write(enhancedFrame);
    }

    cap.release();
    writer.release();

    cout << "Processamento concluído! Vídeo salvo em: " << outputVideoPath << endl;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        cout << "Uso: " << argv[0] << " <input_video> <output_video> <model_path> <model_name> <scale>" << endl;
        return -1;
    }

    string inputVideoPath = argv[1];
    string outputVideoPath = argv[2];
    string modelPath = argv[3];
    string modelName = argv[4]; // ex: "edsr", "fsrcnn", etc.
    int scale = stoi(argv[5]);

    processVideo(inputVideoPath, outputVideoPath, modelPath, modelName, scale);

    return 0;
}