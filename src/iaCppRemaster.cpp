#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <thread>
#include <mutex>

using namespace cv;
using namespace dnn_superres;
using namespace std;

void enhanceFrame(const Mat &inputFrame, Mat &outputFrame, DnnSuperResImpl &sr)
{
    // Aplicar super-resolução ao quadro
    sr.upsample(inputFrame, outputFrame);
}

void removeNoise(const Mat &inputFrame, Mat &outputFrame)
{
    // Aplicar filtro bilateral para remoção de ruído
    bilateralFilter(inputFrame, outputFrame, 9, 75, 75);
}

void interpolateFrames(const Mat &frame1, const Mat &frame2, vector<Mat> &interpolatedFrames, int numInterpolatedFrames)
{
    for (int i = 1; i <= numInterpolatedFrames; ++i)
    {
        float alpha = static_cast<float>(i) / (numInterpolatedFrames + 1);
        Mat interpolatedFrame;
        addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0.0, interpolatedFrame);
        interpolatedFrames.push_back(interpolatedFrame);
    }
}

void processVideo(const string &inputVideoPath, const string &outputVideoPath, DnnSuperResImpl &sr, int scale, int targetFPS)
{
    VideoCapture cap(inputVideoPath);
    if (!cap.isOpened())
    {
        cerr << "Erro ao abrir o vídeo: " << inputVideoPath << endl;
        return;
    }

    // Obter parâmetros do vídeo original
    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));

    // Calcular o número de quadros interpolados necessários
    int numInterpolatedFrames = static_cast<int>((targetFPS / fps) - 1);

    // Configurar o vídeo de saída
    VideoWriter writer(outputVideoPath, fourcc, targetFPS, Size(frameWidth * scale, frameHeight * scale));
    if (!writer.isOpened())
    {
        cerr << "Erro ao criar o vídeo de saída: " << outputVideoPath << endl;
        return;
    }

    Mat frame, enhancedFrame, denoisedFrame, prevFrame;
    bool firstFrame = true;
    mutex mtx;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Processar o quadro em uma thread separada
        thread processingThread([&]()
                                {
            // Remover ruído do quadro
            removeNoise(frame, denoisedFrame);

            // Aplicar super-resolução ao quadro sem ruído
            enhanceFrame(denoisedFrame, enhancedFrame, sr);

            lock_guard<mutex> lock(mtx);
            if (!firstFrame) {
                vector<Mat> interpolatedFrames;
                interpolateFrames(prevFrame, enhancedFrame, interpolatedFrames, numInterpolatedFrames);
                for (const auto& interpolatedFrame : interpolatedFrames) {
                    writer.write(interpolatedFrame);
                }
            }

            writer.write(enhancedFrame);
            prevFrame = enhancedFrame.clone();
            firstFrame = false; });
        processingThread.join();
    }
}

int main(int argc, char **argv)
{
    if (argc < 8)
    {
        cerr << "Uso: " << argv[0] << " <inputVideoPath> <outputVideoPath> <modelPath> <modelName> <scale> <arquitetura> <targetFPS>" << endl;
        return -1;
    }

    string inputVideoPath = argv[1];
    string outputVideoPath = argv[2];
    string modelPath = argv[3];
    string modelName = argv[4];
    int scale = stoi(argv[5]);
    string arquitetura = argv[6];
    int targetFPS = stoi(argv[7]);

    // Inicializar o modelo de super-resolução uma vez
    DnnSuperResImpl sr;
    sr.readModel(modelPath);
    sr.setModel(modelName, scale);

    // Configurar para usar GPU ou CPU
    if (arquitetura == "gpu")
    {
        sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (arquitetura == "cpu")
    {
        sr.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        sr.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    else
    {
        cerr << "Opção inválida para arquitetura: " << arquitetura << ". Use 'cpu' ou 'gpu'." << endl;
        return -1;
    }

    processVideo(inputVideoPath, outputVideoPath, sr, scale, targetFPS);

    return 0;
}