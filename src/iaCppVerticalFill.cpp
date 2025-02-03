#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <vector>

class OpenVINOModel {
public:
    OpenVINOModel(const std::string& modelPath) {
        try {
            // Carregar o modelo OpenVINO
            core = ov::Core();
            model = core.read_model(modelPath);
            compiledModel = core.compile_model(model, "CPU");
        } catch (const std::exception& e) {
            std::cerr << "Erro ao carregar o modelo OpenVINO: " << e.what() << std::endl;
            exit(1);
        }
    }

    cv::Mat run(cv::Mat& input) {
        // Converta o frame do OpenCV para o formato OpenVINO
        ov::Tensor inputTensor = ov::Tensor(ov::element::f32, {1, 3, input.rows, input.cols}, input.data);

        // Executar o modelo OpenVINO
        auto outputTensor = compiledModel.forward({inputTensor}).get_tensor();
        cv::Mat outputImage(input.rows, input.cols, CV_32FC3, outputTensor.data());

        // Processar a saída e retornar a imagem
        cv::Mat result;
        cv::resize(outputImage, result, input.size(), 0, 0, cv::INTER_CUBIC);
        return result;
    }

private:
    ov::Core core;
    ov::Model model;
    ov::CompiledModel compiledModel;
};

// Função para redimensionar o vídeo horizontal para vertical e preencher as áreas faltantes
cv::Mat processFrame(cv::Mat& frame, OpenVINOModel& model) {
    int targetWidth = frame.cols;
    int targetHeight = frame.rows * 2; // Duplo da altura para virar para vertical

    // Criar uma imagem em branco para preencher o lado esquerdo
    cv::Mat newFrame(targetHeight, targetWidth, frame.type(), cv::Scalar(0, 0, 0));

    // Redimensionar o quadro original para o topo da nova imagem
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(targetWidth, frame.rows), 0, 0, cv::INTER_CUBIC);
    resizedFrame.copyTo(newFrame(cv::Rect(0, 0, resizedFrame.cols, resizedFrame.rows)));

    // Usar o modelo de IA para preencher a parte inferior da imagem (simulação com IA)
    cv::Mat filledFrame = model.run(newFrame);

    return filledFrame;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Uso: ./iaCppVerticalFill <video_input_path> <video_output_path> <modelo_openvino>" << std::endl;
        return -1;
    }

    std::string inputVideoPath = argv[1];
    std::string outputVideoPath = argv[2];
    std::string modelPath = argv[3];

    // Inicializar o modelo OpenVINO
    OpenVINOModel model(modelPath);

    // Abrir o vídeo de entrada
    cv::VideoCapture cap(inputVideoPath);
    if (!cap.isOpened()) {
        std::cerr << "Erro ao abrir o vídeo: " << inputVideoPath << std::endl;
        return -1;
    }

    // Obter as propriedades do vídeo de entrada
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frameRate = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // Criar o escritor de vídeo para o output
    cv::VideoWriter writer;
    writer.open(outputVideoPath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frameRate, cv::Size(frameWidth, frameHeight * 2)); // Vertical double size

    if (!writer.isOpened()) {
        std::cerr << "Erro ao abrir o arquivo de saída: " << outputVideoPath << std::endl;
        return -1;
    }

    // Processar o vídeo quadro a quadro
    cv::Mat frame;
    while (cap.read(frame)) {
        // Processar o quadro
        cv::Mat processedFrame = processFrame(frame, model);

        // Escrever o quadro processado no arquivo de saída
        writer.write(processedFrame);
    }

    // Fechar os fluxos de vídeo
    cap.release();
    writer.release();

    std::cout << "Processamento concluído. O vídeo foi salvo em " << outputVideoPath << std::endl;

    return 0;
}