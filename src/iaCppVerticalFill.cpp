#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace ov;

const string INPAINT_MODEL = "lama.xml";
const string SUPER_RES_MODEL = "realesrgan.xml";

// Estrutura para gerenciar modelos OpenVINO
struct OpenVINOModel {
    Core ie;
    CompiledModel compiled_model;
    InferRequest infer_request;

    OpenVINOModel(const string& modelPath) {
        compiled_model = ie.compile_model(modelPath, "CPU");
        infer_request = compiled_model.create_infer_request();
    }

    Mat run(Mat& input) {
        Mat resized;
        resize(input, resized, Size(512, 512), 0, 0, INTER_CUBIC);
        resized.convertTo(resized, CV_32F, 1.0 / 255);

        auto input_tensor = infer_request.get_input_tensor();
        float* data = input_tensor.data<float>();
        memcpy(data, resized.data, resized.total() * sizeof(float));

        infer_request.infer();

        auto output_tensor = infer_request.get_output_tensor();
        Mat result(512, 512, CV_32F, output_tensor.data<float>());
        result.convertTo(result, CV_8U, 255);
        resize(result, input.size(), 0, 0, INTER_CUBIC);
        return result;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Uso: " << argv[0] << " <input_video> <output_video>" << endl;
        return -1;
    }

    string inputVideo = argv[1];
    string outputVideo = argv[2];

    VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        cout << "Erro ao abrir o vídeo: " << inputVideo << endl;
        return -1;
    }

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);

    int newWidth = height * 9 / 16;
    VideoWriter writer(outputVideo, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(newWidth, height));

    if (!writer.isOpened()) {
        cout << "Erro ao abrir o arquivo de saída: " << outputVideo << endl;
        return -1;
    }

    // Criar modelos OpenVINO uma única vez
    OpenVINOModel inpaintingModel(INPAINT_MODEL);
    OpenVINOModel superResModel(SUPER_RES_MODEL);

    Mat frame, croppedFrame, leftMask, rightMask, filledLeft, filledRight, finalFrame;
    Rect roi((width - newWidth) / 2, 0, newWidth, height);

    #pragma omp parallel
    {
        while (cap.read(frame)) {
            croppedFrame = frame(roi);

            // Criar máscaras laterais
            leftMask = frame(Rect(0, 0, roi.x, height));
            rightMask = frame(Rect(roi.x + newWidth, 0, roi.x, height));

            // Paralelizar IA nas laterais
            #pragma omp parallel sections
            {
                #pragma omp section
                filledLeft = inpaintingModel.run(leftMask);

                #pragma omp section
                filledRight = inpaintingModel.run(rightMask);
            }

            // Combinar imagens com operações vetorizadas
            parallel_for_(Range(0, height), [&](const Range& range) {
                for (int i = range.start; i < range.end; i++) {
                    memcpy(finalFrame.ptr(i, 0), filledLeft.ptr(i, 0), roi.x * frame.elemSize());
                    memcpy(finalFrame.ptr(i, roi.x), croppedFrame.ptr(i, 0), newWidth * frame.elemSize());
                    memcpy(finalFrame.ptr(i, roi.x + newWidth), filledRight.ptr(i, 0), roi.x * frame.elemSize());
                }
            });

            // Super-resolução
            finalFrame = superResModel.run(finalFrame);

            writer.write(finalFrame);
        }
    }

    cap.release();
    writer.release();
    cout << "Vídeo processado com sucesso: " << outputVideo << endl;
    return 0;
}