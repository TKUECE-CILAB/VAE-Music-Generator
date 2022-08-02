#%%
from utils import json2midi
from inference import inference

import os
import json
from tqdm import tqdm

#%%
if __name__ == '__main__':

    output_num = 5

    weight_base = os.path.join("model", "weights")
    out_midi_base = "json_out" 

    emotion_name = ["Happy", "Angry", "Relaxed", "Sad"]

    weight_folders = [
        os.path.join(weight_base, emotion_name[0]),
        os.path.join(weight_base, emotion_name[1]),
        os.path.join(weight_base, emotion_name[2]),
        os.path.join(weight_base, emotion_name[3]),
    ]

    json_folders = [
        os.path.join(out_midi_base, emotion_name[0]),
        os.path.join(out_midi_base, emotion_name[1]),
        os.path.join(out_midi_base, emotion_name[2]),
        os.path.join(out_midi_base, emotion_name[3]),
    ]


    for i, folder in enumerate(weight_folders):
            weight = os.path.join(folder, "latest_weight.pth")

            for _ in tqdm(range(output_num), desc=emotion_name[i]):

                json_output = json.loads(inference(weight))


                valence = json_output['valence']
                energy = json_output['energy']

                out_name = "V_{} E_{}".format(valence, energy) +  ".json"
                out_path = os.path.join(json_folders[i], out_name)

                with open(out_path, "w") as f:
                    json.dump(json_output, f)

#%%