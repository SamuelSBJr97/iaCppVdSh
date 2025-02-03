import cv2
import os
import whisper
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

# Inicializa o modelo Whisper para transcrição
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

def generate_new_video(original_video_path, transcript, prompt, output_path="output_video.mp4"):
    """
    Gera um novo vídeo baseado no original, com legendas da transcrição e texto do prompt.
    """
    # Carregar o vídeo original usando MoviePy
    print(f"Gerando novo vídeo: {output_path}")
    video = VideoFileClip(original_video_path)
    duration = video.duration

    # Criar legendas a partir da transcrição
    subtitle_clip = TextClip(
        transcript,
        fontsize=24,
        color='white',
        bg_color='black',
        size=(video.w, 50),
        method='caption'
    ).set_position(('center', video.h - 60)).set_duration(duration)

    # Criar um texto adicional baseado no prompt
    prompt_clip = TextClip(
        f"Prompt: {prompt}",
        fontsize=24,
        color='yellow',
        bg_color='black',
        size=(video.w, 50),
        method='caption'
    ).set_position(('center', 20)).set_duration(duration)

    # Compor o vídeo com as legendas e o texto do prompt
    final_video = CompositeVideoClip([video, subtitle_clip, prompt_clip])

    # Exportar o vídeo
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

def analyze_film_and_generate_new(video_path, prompt):
    """
    Processa o vídeo para extrair frames, transcrever áudio e gerar um novo vídeo.
    """
    try:
        # Extrair frames e taxa de quadros
        frames, frame_rate = extract_frames_from_video(video_path)

        # Extrair e transcrever áudio
        audio_path = extract_audio_from_video(video_path)
        transcript = transcribe_audio(audio_path)

        # Gerar um novo vídeo com transcrição e texto do prompt
        output_video_path = "new_video_with_transcriptions.mp4"
        generate_new_video(video_path, transcript, prompt, output_video_path)

        print("\nProcessamento concluído!")
        print(f"Novo vídeo gerado: {output_video_path}")
        return transcript, output_video_path
    except Exception as e:
        print(f"Erro ao processar o vídeo: {e}")
        return None, None

if __name__ == "__main__":
    # Caminho do vídeo de entrada e texto do prompt
    video_path = "input_movie.mkv"
    prompt = "Este é um exemplo de como criar um novo filme com IA."

    # Processa o vídeo, gera transcrição e cria um novo vídeo
    transcript, new_video = analyze_film_and_generate_new(video_path, prompt)

    if transcript is not None and new_video is not None:
        print("\nTranscrição do áudio:")
        print(transcript)
        print(f"Novo vídeo gerado: {new_video}")
    else:
        print("Falha no processamento do vídeo.")
