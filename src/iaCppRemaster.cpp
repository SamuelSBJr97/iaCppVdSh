#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <immintrin.h> // Para intrínsecos SIMD
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace dnn_superres;
using namespace std;

void enhanceFrame(const Mat &inputFrame, Mat &outputFrame, DnnSuperResImpl &sr)
{
    sr.upsample(inputFrame, outputFrame);
}

void removeNoise(const Mat &inputFrame, Mat &outputFrame)
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
    }
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

void processFrame(const Mat &frame, Mat &enhancedFrame, DnnSuperResImpl &sr)
{
    Mat denoisedFrame;
    removeNoise(frame, denoisedFrame);
    enhanceFrame(denoisedFrame, enhancedFrame, sr);
}

string generateFilename(const string &baseName, int frameNumber)
{
    stringstream ss;
    ss << baseName << "_frame_" << setw(6) << setfill('0') << frameNumber << ".mp4";
    return ss.str();
}

void savePartialVideo(const Mat &frame, const string &baseName, int frameNumber, int fourcc, double fps, int width, int height)
{
    string filename = generateFilename(baseName, frameNumber);
    VideoWriter writer(filename, fourcc, fps, Size(width, height));
    if (!writer.isOpened())
    {
        cerr << "Erro ao criar o vídeo parcial: " << filename << endl;
        return;
    }
    writer.write(frame);
    cout << "Salvando vídeo parcial: " << filename << endl;
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

    cout << "Configurando vídeo de saída: " << outputVideoPath << endl;
    VideoWriter writer(outputVideoPath, fourcc, targetFPS, Size(frameWidth * scale, frameHeight * scale));
    if (!writer.isOpened())
    {
        cerr << "Erro ao criar o vídeo de saída: " << outputVideoPath << endl;
        return;
    }

    Mat frame, enhancedFrame, prevFrame;
    bool firstFrame = true;
    mutex mtx;
    queue<Mat> frameQueue;
    condition_variable cv;
    bool done = false;
    int frameNumber = 0;

    auto worker = [&]()
    {
        while (true)
        {
            Mat frame;
            {
                unique_lock<mutex> lock(mtx);
                cv.wait(lock, [&]()
                        { return !frameQueue.empty() || done; });
                if (done && frameQueue.empty())
                    break;
                frame = frameQueue.front();
                frameQueue.pop();
            }
            Mat enhancedFrame;
            processFrame(frame, enhancedFrame, sr);
            {
                lock_guard<mutex> lock(mtx);
                if (!firstFrame)
                {
                    vector<Mat> interpolatedFrames;
                    interpolateFrames(prevFrame, enhancedFrame, interpolatedFrames, numInterpolatedFrames);
                    for (const auto &interpolatedFrame : interpolatedFrames)
                    {
                        writer.write(interpolatedFrame);
                        savePartialVideo(interpolatedFrame, outputVideoPath, frameNumber++, fourcc, targetFPS, frameWidth * scale, frameHeight * scale);
                    }
                }
                writer.write(enhancedFrame);
                savePartialVideo(enhancedFrame, outputVideoPath, frameNumber++, fourcc, targetFPS, frameWidth * scale, frameHeight * scale);
                prevFrame = enhancedFrame.clone();
                firstFrame = false;
            }
        }
    };

    cout << "Iniciando threads de processamento..." << endl;
    vector<thread> workers;
    int numThreads = thread::hardware_concurrency();
    for (int i = 0; i < numThreads; ++i)
    {
        workers.emplace_back(worker);
    }

    cout << "Processando quadros do vídeo..." << endl;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;
        {
            lock_guard<mutex> lock(mtx);
            frameQueue.push(frame);
        }
        cv.notify_one();
        cout << "Quadro lido e adicionado à fila: " << frameNumber << endl;
    }

    {
        lock_guard<mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();

    for (auto &worker : workers)
    {
        worker.join();
    }

    cout << "Processamento concluído. Vídeo salvo em: " << outputVideoPath << endl;
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