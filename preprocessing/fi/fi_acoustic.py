import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from exordium.audio.smile import OpensmileWrapper
from exordium.audio.wav2vec import Wav2vec2Wrapper
from exordium.audio.io import video2audio

DB = Path("data/db_processed/fi")
DB_VIDEOS = DB / "videos"
DB_AUDIOS = DB / "Audio"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to preprocess FI acoustic features.")
    parser.add_argument('--start',  type=int, default=0, help='participant id slice start')
    parser.add_argument('--end',    type=int, default=10000, help='participant id slice end')
    args = parser.parse_args()

    for converted_video_path in tqdm(sorted(list(DB_VIDEOS.glob("**/*.mp4")))):
        audio_path = DB_AUDIOS / converted_video_path.parent.name / f'{converted_video_path.stem}.wav'
        if audio_path.exists(): continue
        video2audio(
            converted_video_path,
            audio_path,
            sr=16000
        )

    wav2vec_extractor = Wav2vec2Wrapper()
    opensmile_extractor = OpensmileWrapper('egemaps', 'lld')

    audio_paths = sorted(list(DB_AUDIOS.glob("**/*.wav")))[args.start:args.end]
    print('Number of audios:', len(audio_paths))

    for v, audio_path in tqdm(enumerate(audio_paths), total=len(audio_paths), desc='Audios'):
        start_time = time.time()
        video_id = audio_path.stem

        try:
            wav2vec_feature =wav2vec_extractor.audio_to_feature(audio_path, output_path=DB / 'wav2vec2' / f'{video_id}.npy')
            smile_feature = opensmile_extractor.audio_to_feature(audio_path, output_path=DB / 'egemaps_lld' / f'{video_id}.npy')
        except Exception as e:
            with open(DB / "fi_skip_audio.txt", "a") as f:
                f.write(f'{video_id},{e}\n')

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:03f} seconds")