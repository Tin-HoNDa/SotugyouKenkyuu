import av
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoModel
import pandas as pd
import gensim.models.keyedvectors as word2vec
from scipy import spatial
import matplotlib.pyplot as plt

def min_max_normalization(l):
    #l_min = min(l)
    #l_max = max(l)
    l_min = 1
    l_max = 5
    return [(i - l_min) / (l_max - l_min) for i in l]

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def average_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def load_model():
    model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    return model

def avg_feature_vector(sentence, model, num_features):
    words = sentence.replace(',', '').replace('.', '').split()
    feature_vec = np.zeros((num_features,), dtype="float32")
    miss = 0
    for word in words:
        try:
            feature_vec = np.add(feature_vec, model[word])
        except KeyError as e:
            miss += 1
            print(e)
        feature_vec = np.divide(feature_vec, len(words)-miss)
    return feature_vec

def sentence_similarity(sentence_1, sentence_2):
    num_features=300
    sentence_1_avg_vector = avg_feature_vector(sentence_1, model, num_features)
    sentence_2_avg_vector = avg_feature_vector(sentence_2, model, num_features)
    return 1 - spatial.distance.cosine(sentence_1_avg_vector, sentence_2_avg_vector)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("init")

# load video
dir_videos = "mp4/"
dir_prompts = "./アンケート/prompts/"

video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
prompt_paths = [os.path.join(dir_prompts, x) for x in os.listdir(dir_prompts)]

for p in video_paths:
    print(p)

print("captioning will starts...")

sequences = []

for i in tqdm(range(len(video_paths))):
    container = av.open(video_paths[i])

    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    gen_kwargs = {
        "min_length": 10,
        "max_length": 20,
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    sequences.append(caption)

print("captioning finished.")

#model = SentenceTransformer("intfloat/multilingual-e5-base")
#tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
#model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("comparing will starts...")

outputs = []

for i in tqdm(range(len(video_paths))):
    prompt_path = prompt_paths[i]
    sequence = sequences[i]

    sequence = 'query: ' + str(sequence)
    text = 'passage: ' + str(read_text_file(prompt_path))

    input_texts = [sequence, sequence, text, text]
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    output = model(**batch_dict)
    embeddings = average_pool(output.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:2] @ embeddings[2:].T)
    output = scores[0][0]
    output = output.detach().numpy()
    outputs.append(output)

#model = load_model()
#
#for i in tqdm(range(len(video_paths))):
#    prompt_path = prompt_paths[i]
#    sequence = sequences[i]
#
#    text = str(read_text_file(prompt_path))
#
#    score = sentence_similarity(sequence, text)
#
#    outputs.append(score)

print("comparing finished.")

for i in range(len(prompt_paths)):
    print(f"{i + 1}:{sequences[i]}, score:{outputs[i]}")

print(f"Final average score: {sum(outputs)/len(outputs)}, Total videos: {len(outputs)}")

csv = pd.read_csv("./Ground_Truth.csv")
ground_score = []

print("compare to human evaluation")

for i in range(len(prompt_paths)):
    index = 'q' + str(3 * i + 2)
    ground_score.append(csv[index].mean())

ground_score = min_max_normalization(ground_score)

S_scores = pd.Series(outputs)
S_ground_score = pd.Series(ground_score)

print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")

plt.scatter(ground_score, outputs)
plt.xlabel("Ground-truth")
plt.ylabel("Our method")
plt.plot(ground_score, np.poly1d(np.polyfit(ground_score, outputs, 1))(ground_score))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig("卒論/vid_corr.png")

print("compare to general evaluation")

ground_score = []

for i in range(len(prompt_paths)):
    index = 'q' + str(3 * i + 3)
    ground_score.append(csv[index].mean())

ground_score = min_max_normalization(ground_score)

S_scores = pd.Series(outputs)
S_ground_score = pd.Series(ground_score)

print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")

print()
print()

print("human")

for i in range(len(csv)):
    c = csv[i:i+1]
    ground_score = []
    for j in range(len(prompt_paths)):
        index = 'q' + str(3 * j + 2)
        ground_score.append(c[index].mean())
    ground_score = min_max_normalization(ground_score)
    S_ground_score = pd.Series(ground_score)
    print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}, Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")
