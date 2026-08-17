from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pickle
from exordium.text.roberta import RobertaWrapper
from exordium.text.bert import BertWrapper


# 自动切换到项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
os.chdir(PROJECT_ROOT)

DB = Path("data/db/FI")
DB_PROCESSED = Path("data/db_processed/fi")
DB_PROCESSED_TEXT = DB_PROCESSED / "text"
TRANSCRIPT_DIR = DB / 'gt'


def load_transcript(path: str | Path) -> dict:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    transcript_dict = load_transcript(TRANSCRIPT_DIR / 'transcription_training.pkl') \
        | load_transcript(TRANSCRIPT_DIR / 'transcription_validation.pkl') \
        | load_transcript(TRANSCRIPT_DIR / 'transcription_test.pkl')

    bert = BertWrapper()
    roberta = RobertaWrapper()

    dict_roberta, dict_bert = {}, {}
    for video_name, transcript in tqdm(transcript_dict.items(), total=len(transcript_dict)):
        video_id = video_name[:-4]
        text = transcript.lower()

        feature_roberta = roberta(text)
        feature_roberta = feature_roberta.squeeze(0).numpy()
        dict_roberta[video_id] = feature_roberta

        feature_bert = bert(text)
        feature_bert = feature_bert.squeeze(0).numpy()
        dict_bert[video_id] = feature_bert
    
    DB_PROCESSED_TEXT.mkdir(parents=True, exist_ok=True)

    with open(DB_PROCESSED_TEXT / 'fi_roberta.pkl', 'wb') as f:
        pickle.dump(dict_roberta, f)

    with open(DB_PROCESSED_TEXT / 'fi_bert.pkl', 'wb') as f:
        pickle.dump(dict_bert, f)