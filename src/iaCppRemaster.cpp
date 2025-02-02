#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn_superres.hpp>
#include <thread>
#include <vector>

using namespace std;
using namespace cv;
using namespace dnn_superres;

void enhanceFrame(const Mat &inputFrame, Mat &outputFrame, DnnSuperResImpl &sr)
{
    sr.upsample(inputFrame, outputFrame);
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
    VideoWriter writer(outputVideoPath, fourcc, targetFps, Size(frameWidth * scale, frameHeight * scale));
    if (!writer.isOpened())
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

    // Thread para processar quadros
    auto workerThread = [&]()
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
            enhanceFrame(localFrame, enhanced, sr);

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
            Mat localEnhanced;
            {
                unique_lock<mutex> lock(queueMutex);
                cv.wait(lock, [&]()
                        { return !enhancedQueue.empty() || (finishedReading && framesQueue.empty()); });
                if (enhancedQueue.empty() && finishedReading && framesQueue.empty())
                    break;

                localEnhanced = enhancedQueue.front();
                enhancedQueue.pop();
            }

            // Escrever o mesmo quadro múltiplas vezes para ajustar o FPS
            for (int i = 0; i < frameRepeat; ++i)
            {
                writer.write(localEnhanced);
            }
        }
    };

    thread reader(readerThread);
    workers.emplace_back(workerThread);
    workers.emplace_back(workerThread);
    thread writer(writerThread);

    reader.join();
    for (auto &worker : workers)
    {
        worker.join();
    }
    writer.join();

    cap.release();
    writer.release();

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