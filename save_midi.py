
# %%
from utils import json2midi

import os
import json
from tqdm import tqdm

# %%
if __name__ == '__main__':

    midi_base = "midi_out"
    json_base = "json_out"

    emotion_name = ["Happy", "Angry", "Relaxed", "Sad"]

    json_folders = [
        os.path.join(json_base, emotion_name[0]),
        os.path.join(json_base, emotion_name[1]),
        os.path.join(json_base, emotion_name[2]),
        os.path.join(json_base, emotion_name[3]),
    ]

    midi_folders = [
        os.path.join(midi_base, emotion_name[0]),
        os.path.join(midi_base, emotion_name[1]),
        os.path.join(midi_base, emotion_name[2]),
        os.path.join(midi_base, emotion_name[3]),
    ]

    for i, emo_folder in enumerate(json_folders):
        for json_data in tqdm(os.listdir(emo_folder), desc=emotion_name[i]):

            json_path = os.path.join(emo_folder, json_data)

            midi = json2midi.decode(json_path, main_program=25, chord_program=4)
            out_name = json_data + ".mid"
            out_path = os.path.join(midi_folders[i], out_name)
            midi.write(out_path)

# %%