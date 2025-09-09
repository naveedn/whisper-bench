import mlx_whisper

def main():
    print("Hello from mlx-whisper-test!")
    result = mlx_whisper.transcribe('../audio-files-wav/2-zaboombafool.wav')
    print(result["text"])


if __name__ == "__main__":
    main()
