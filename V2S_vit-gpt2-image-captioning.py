from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoModel
import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import os
import time
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import gensim.models.keyedvectors as word2vec
from scipy import spatial

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def min_max_normalization(l):
    #l_min = min(l)
    #l_max = max(l)
    l_min = 1
    l_max = 5
    return [(((i - l_min) / (l_max - l_min)) - 0.5) * 2 for i in l]

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

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

def be_binary(row):
    row = row.tolist()
    binary = []
    for i in range(len(row)):
        if row[i] < 3:
            binary.append(-1)
        elif row[i] > 3:
            binary.append(1)
    binary = pd.Series(binary)
    return binary

print("init")

#dir_videos = "./アンケート/videos/"
dir_videos = "splitted/enquete/"
dir_prompts = "./アンケート/prompts/"

video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
prompt_paths = [os.path.join(dir_prompts, x) for x in os.listdir(dir_prompts)]

print("captioning will starts...")

sequences = []

test_num = len(video_paths)
print(f"test_num:{test_num}")

for i in tqdm(range(len(video_paths))):
    video_path = video_paths[i]
    sequence = predict_step([video_path])
    sequences.append(sequence[0])

print("captioning finished.")

model = SentenceTransformer("intfloat/multilingual-e5-base")
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

#model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
#model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("comparing will starts...")

#outputs = []

#for i in tqdm(range(len(video_paths))):
#    prompt_path = prompt_paths[i]
#    sequence = sequences[i]
#
#    sequence = 'query: ' + str(sequence)
#    text = 'passage: ' + str(read_text_file(prompt_path))
#
#    input_texts = [sequence, sequence, text, text]
#    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
#    output = model(**batch_dict)
#    embeddings = average_pool(output.last_hidden_state, batch_dict['attention_mask'])
#    embeddings = F.normalize(embeddings, p=2, dim=1)
#    scores = (embeddings[:2] @ embeddings[2:].T)
#    output = scores[0][0]
#    output = output.detach().numpy()
#    outputs.append(output)
#
#model = load_model()
#
#for i in tqdm(range(len(video_paths))):
#    prompt_path = prompt_paths[i]
#    sequence = sequences[i]
#
#    text = str(read_text_file(prompt_path))
#
#    score = sentence_similarity(str(sequence), text)
#
#    outputs.append(score)
#
#for i in range(len(prompt_paths)):
#    print(f"{i + 1}:{sequences[i]}, score:{outputs[i]}")

min_outputs = []
max_outputs = []
mean_outputs = []

for i in tqdm(range(len(prompt_paths))):
    prompt_path = prompt_paths[i]
    scores = []

    for j in range(16):
        sequence = sequences[i * 16 + j]

        sequence = 'query: ' + str(sequence)
        text = 'passage: ' + str(read_text_file(prompt_path))

        input_texts = [sequence, sequence, text, text]
        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        output = model(**batch_dict)
        embeddings = average_pool(output.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        output = (embeddings[:2] @ embeddings[2:].T)

        output = output[0][0]
        output = output.detach().numpy()
        scores.append(output)
    mean_outputs.append(sum(scores) / len(scores))
    max_outputs.append(max(scores))
    min_outputs.append(min(scores))

#model = load_model()
#
#for i in tqdm(range(len(prompt_paths))):
#    prompt_path = prompt_paths[i]
#    text = str(read_text_file(prompt_path))
#    scores = []
#    for j in range(16):
#        sequence = sequences[i * 16 + j]
#        score = sentence_similarity(str(sequence), text)
#        scores.append(score)
#    outputs.append(sum(scores) / len(scores))
#    #outputs.append(max(scores))
#    #outputs.append(min(scores))

print("mean")
for i in range(len(prompt_paths)):
    print(f"{i + 1}:")
    for j in range(16):
        print(sequences[i * 16 + j])
    print(f"score:{mean_outputs[i]}")
print("max")
for i in range(len(prompt_paths)):
    print(f"{i + 1}:")
    for j in range(16):
        print(sequences[i * 16 + j])
    print(f"score:{max_outputs[i]}")
print("min")
for i in range(len(prompt_paths)):
    print(f"{i + 1}:")
    for j in range(16):
        print(sequences[i * 16 + j])
    print(f"score:{min_outputs[i]}")

print("comparing finished.")

csv = pd.read_csv("./Ground_Truth.csv")
ground_score = []

for i in range(len(prompt_paths)):
    index = 'q' + str(3 * i + 2)
    row = be_binary(csv[index])
    ground_score.append(row.mean())
    print(f"{i+1}:{row.mean()}")

ground_score = min_max_normalization(ground_score)

print("Final average score")
print(f"mean: {sum(mean_outputs)/len(mean_outputs)}, Total videos: {len(mean_outputs)}")
S_scores = pd.Series(mean_outputs)
S_ground_score = pd.Series(ground_score)
print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")
print(f"Spearman's : {S_ground_score.corr(S_scores, method='spearman')}")
print(f"Kendall's : {S_ground_score.corr(S_scores, method='kendall')}")

print(f"max: {sum(max_outputs)/len(max_outputs)}, Total videos: {len(max_outputs)}")
S_scores = pd.Series(max_outputs)
S_ground_score = pd.Series(ground_score)
print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")
print(f"Spearman's : {S_ground_score.corr(S_scores, method='spearman')}")
print(f"Kendall's : {S_ground_score.corr(S_scores, method='kendall')}")

print(f"min: {sum(min_outputs)/len(min_outputs)}, Total videos: {len(min_outputs)}")
S_scores = pd.Series(min_outputs)
S_ground_score = pd.Series(ground_score)
print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")
print(f"Spearman's : {S_ground_score.corr(S_scores, method='spearman')}")
print(f"Kendall's : {S_ground_score.corr(S_scores, method='kendall')}")
