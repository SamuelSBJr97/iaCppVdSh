#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <iostream>
#include <immintrin.h> // Para intrínsecos SIMD
#include <sstream>
#include <iomanip>
#include <thread>
#include <condition_variable>
#include <vector>
#include <queue>
#include <mutex>

using namespace cv;
using namespace dnn_superres;
using namespace std;

string generateFilename(const string &baseName, int frameNumber)
{
    stringstream ss;
    ss << baseName << "_frame_" << setw(6) << setfill('0') << frameNumber << ".png";
    return ss.str();
}

void enhanceFrame(const Mat &inputFrame, Mat &outputFrame, DnnSuperResImpl &sr)
{
    // Aplicar super-resolução diretamente
    sr.upsample(inputFrame, outputFrame);
    // outputFrame = inputFrame.clone(); // Apenas clonar o quadro de entrada para saída
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

void processVideo(const string &inputVideoPath, const string &outputVideoPath, const string &modelPath, const string &modelName, int scale, double targetFps)
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
    double originalFps = cap.get(CAP_PROP_FPS);
    int fourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));

    // Configurar o vídeo de saída com o novo FPS
    VideoWriter videoWriter(outputVideoPath, fourcc, targetFps, Size(frameWidth * scale, frameHeight * scale));
    if (!videoWriter.isOpened())
    {
        cerr << "Erro ao criar o vídeo de saída: " << outputVideoPath << endl;
        return;
    }

    // Inicializar o modelo de super-resolução fora do loop para eficiência
    DnnSuperResImpl sr;
    sr.readModel(modelPath);
    sr.setModel(modelName, scale);

    Mat frame, enhancedFrame;
    vector<thread> workers;
    queue<Mat> framesQueue;
    queue<Mat> enhancedQueue;
    mutex queueMutex;
    condition_variable cv;

    bool finishedReading = false;

    // Thread para leitura de quadros
    auto readerThread = [&]()
    {
        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                lock_guard<mutex> lock(queueMutex);
                finishedReading = true;
                cv.notify_all();
                break;
            }
            {
                lock_guard<mutex> lock(queueMutex);
                framesQueue.push(frame.clone());
            }
            cv.notify_all();
        }
    };

    // Thread para processamento de quadros
    auto processorThread = [&]()
    {
        while (true)
        {
            Mat localFrame;
            {
                unique_lock<mutex> lock(queueMutex);
                cv.wait(lock, [&]()
                        { return !framesQueue.empty() || finishedReading; });
                if (framesQueue.empty() && finishedReading)
                    break;

                localFrame = framesQueue.front();
                framesQueue.pop();
            }

            Mat enhanced;
            processFrame(localFrame, enhanced, sr);

            {
                lock_guard<mutex> lock(queueMutex);
                enhancedQueue.push(enhanced);
            }
            cv.notify_all();
        }
    };

    // Thread para salvar quadros
    auto writerThread = [&]()
    {
        int frameRepeat = static_cast<int>(targetFps / originalFps);
        while (true)
        {
            Mat localFrame;
            {
                unique_lock<mutex> lock(queueMutex);
                cv.wait(lock, [&]()
                        { return !enhancedQueue.empty() || finishedReading; });
                if (enhancedQueue.empty() && finishedReading)
                    break;

                localFrame = enhancedQueue.front();
                enhancedQueue.pop();
            }

            for (int i = 0; i < frameRepeat; ++i)
            {
                videoWriter.write(localFrame);
            }
        }
    };

    // Iniciar threads
    thread reader(readerThread);
    thread processor(processorThread);
    thread writer(writerThread);

    // Aguardar threads
    reader.join();
    processor.join();
    writer.join();

    cap.release();
    videoWriter.release();

    cout << "Processamento concluído! Vídeo salvo em: " << outputVideoPath << endl;
}

int main(int argc, char **argv)
{
    if (argc != 7)
    {
        cout << "Uso: " << argv[0] << " <input_video> <output_video> <model_path> <model_name> <scale> <target_fps>" << endl;
        return -1;
    }

    string inputVideoPath = argv[1];
    string outputVideoPath = argv[2];
    string modelPath = argv[3];
    string modelName = argv[4]; // ex: "realesrgan", "fsrcnn", etc.
    int scale = stoi(argv[5]);
    double targetFps = stod(argv[6]);

    processVideo(inputVideoPath, outputVideoPath, modelPath, modelName, scale, targetFps);

    return 0;
}