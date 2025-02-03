import cv2
import os
import whisper

# Inicializa o modelo Whisper
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_path):
    """
    Transcreve áudio para texto usando Whisper.
    """
    print(f"Transcrevendo o áudio: {audio_path}")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def extract_audio_from_video(video_path, audio_path="audio_temp.wav"):
    """
    Extrai o áudio de um arquivo de vídeo e o salva como WAV.
    """
    print(f"Extraindo áudio do vídeo: {video_path}")
    command = f"ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y"
    os.system(command)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Erro ao extrair áudio do vídeo: {audio_path} não encontrado.")
    return audio_path

def extract_frames_from_video(video_path):
    """
    Extrai frames do vídeo.
    """
    print(f"Extraindo frames do vídeo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

    cap.release()
    print(f"Número total de frames extraídos: {frame_count}")
    return frames, frame_rate

def analyze_film(video_path):
    """
    Processa o vídeo para extrair frames e transcrever o áudio.
    """
    try:
        # Extrair frames
        frames, frame_rate = extract_frames_from_video(video_path)

        # Extrair e transcrever áudio
        audio_path = extract_audio_from_video(video_path)
        transcript = transcribe_audio(audio_path)

        return frames, frame_rate, transcript
    except Exception as e:
        print(f"Erro ao processar o vídeo: {e}")
        return None, None, None

if __name__ == "__main__":
    # Caminho do vídeo de entrada
    video_path = "input_movie.mp4"

    # Processa o vídeo
    frames, frame_rate, transcript = analyze_film(video_path)

    if frames is not None and transcript is not None:
        print("\nTranscrição do áudio:")
        print(transcript)
        print(f"\nNúmero de frames extraídos: {len(frames)}")
        print(f"Taxa de quadros (FPS): {frame_rate}")
    else:
        print("Falha no processamento do vídeo.")
