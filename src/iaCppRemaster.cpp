#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <immintrin.h> // Para intrínsecos SIMD
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace dnn_superres;
using namespace std;

string generateFilename(const string &baseName, int frameNumber)
{
    stringstream ss;
    ss << baseName << "_frame_" << setw(6) << setfill('0') << frameNumber << ".png";
    return ss.str();
}

void savePartialImage(const Mat &frame, const string &baseName, int frameNumber)
{
    string filename = generateFilename(baseName, frameNumber);
    if (!imwrite(filename, frame))
    {
        cerr << "Erro ao salvar a imagem parcial: " << filename << endl;
        return;
    }
    cout << "Salvando imagem parcial: " << filename << endl;
}

void enhanceFrame(const Mat &inputFrame, Mat &outputFrame, DnnSuperResImpl &sr)
{
    // Aplicar super-resolução diretamente
    //sr.upsample(inputFrame, outputFrame);
    cout << "Quadro aprimorado." << endl;
}

void removeNoise(const Mat &inputFrame, Mat &outputFrame, const string &baseName, int frameNumber)
{
    // Usar filtro bilateral com SIMD
    Mat tempFrame;
    bilateralFilter(inputFrame, tempFrame, 9, 75, 75);

    // Aplicar operações SIMD para otimizar a remoção de ruído
    const int channels = tempFrame.channels();
    const int nRows = tempFrame.rows;
    const int nCols = tempFrame.cols * channels;

    outputFrame.create(tempFrame.size(), tempFrame.type());

    for (int i = 0; i < nRows; ++i)
    {
        const uchar *pSrc = tempFrame.ptr<uchar>(i);
        uchar *pDst = outputFrame.ptr<uchar>(i);
        for (int j = 0; j < nCols; j += 16)
        {
            __m128i src = _mm_loadu_si128((__m128i *)&pSrc[j]);
            _mm_storeu_si128((__m128i *)&pDst[j], src);
        }
        cout << "Removendo ruído: linha " << i + 1 << " de " << nRows << endl;
    }

    // Salvar imagem parcial após remover ruído
    savePartialImage(outputFrame, baseName, frameNumber);
}

void interpolateFrames(const Mat &frame1, const Mat &frame2, vector<Mat> &interpolatedFrames, int numInterpolatedFrames)
{
    for (int i = 1; i <= numInterpolatedFrames; ++i)
    {
        float alpha = static_cast<float>(i) / (numInterpolatedFrames + 1);
        Mat interpolatedFrame;
        addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0.0, interpolatedFrame);
        interpolatedFrames.push_back(interpolatedFrame);
        cout << "Interpolação de quadros: " << i << " de " << numInterpolatedFrames << endl;
    }
}

void processFrame(const Mat &frame, Mat &enhancedFrame, DnnSuperResImpl &sr, const string &baseName, int frameNumber)
{
    Mat denoisedFrame;
    removeNoise(frame, denoisedFrame, baseName, frameNumber);
    enhanceFrame(denoisedFrame, enhancedFrame, sr);
}

void displayProgressBar(int current, int total)
{
    int barWidth = 70;
    float progress = (float)current / total;
    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
}

void processVideo(const string &inputVideoPath, const string &outputVideoPath, DnnSuperResImpl &sr, int scale, int targetFPS)
{
    cout << "Abrindo vídeo de entrada: " << inputVideoPath << endl;
    VideoCapture cap(inputVideoPath);
    if (!cap.isOpened())
    {
        cerr << "Erro ao abrir o vídeo: " << inputVideoPath << endl;
        return;
    }

    int frameWidth = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));
    int numInterpolatedFrames = static_cast<int>((targetFPS / fps) - 1);
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    cout << "Configurando vídeo de saída: " << outputVideoPath << endl;
    VideoWriter writer(outputVideoPath, fourcc, targetFPS, Size(frameWidth * scale, frameHeight * scale));
    if (!writer.isOpened())
    {
        cerr << "Erro ao criar o vídeo de saída: " << outputVideoPath << endl;
        return;
    }

    Mat frame, enhancedFrame, prevFrame;
    bool firstFrame = true;
    int frameNumber = 0;

    cout << "Processando quadros do vídeo..." << endl;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        processFrame(frame, enhancedFrame, sr, outputVideoPath, frameNumber);

        if (!firstFrame)
        {
            vector<Mat> interpolatedFrames;
            interpolateFrames(prevFrame, enhancedFrame, interpolatedFrames, numInterpolatedFrames);
            for (int i = 0; i < interpolatedFrames.size(); ++i)
            {
                writer.write(interpolatedFrames[i]);
                savePartialImage(interpolatedFrames[i], outputVideoPath, frameNumber++);
                cout << "Quadro interpolado salvo: " << i + 1 << " de " << interpolatedFrames.size() << endl;
            }
        }

        writer.write(enhancedFrame);
        savePartialImage(enhancedFrame, outputVideoPath, frameNumber++);
        prevFrame = enhancedFrame.clone();
        firstFrame = false;

        cout << "Quadro processado: " << frameNumber << endl;
        displayProgressBar(frameNumber, totalFrames);
    }

    cout << endl
         << "Processamento concluído. Vídeo salvo em: " << outputVideoPath << endl;
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

    cout << "Inicializando modelo de super-resolução..." << endl;
    DnnSuperResImpl sr;
    sr.readModel(modelPath);
    sr.setModel(modelName, scale);

    if (arquitetura == "gpu")
    {
        cout << "Usando GPU para processamento..." << endl;
        sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (arquitetura == "cpu")
    {
        cout << "Usando CPU para processamento..." << endl;
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