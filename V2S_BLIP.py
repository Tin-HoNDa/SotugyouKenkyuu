from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipForConditionalGeneration
import torch
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pandas as pd

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

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
    return [(i - l_min) / (l_max - l_min) for i in l]

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

def average_pool(last_hidden_states: Tensor,attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

print("init")

dir_videos = "./アンケート/videos/"
#dir_videos = "splitted/enquete/"
dir_prompts = "./アンケート/prompts/"

video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
prompt_paths = [os.path.join(dir_prompts, x) for x in os.listdir(dir_prompts)]

print("captioning will starts...")

sequences = []

test_num = len(video_paths)
print(f"test_num:{test_num}")

for i in tqdm(range(len(video_paths))):
    video_path = video_paths[i]
    raw_image = Image.open(video_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to(device)
    sequence = model.generate(**inputs)
    sequence = processor.decode(sequence[0], skip_special_tokens=True)
    sequences.append(sequence)

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

for i in range(len(prompt_paths)):
    print(f"{i + 1}:{sequences[i]}, score:{outputs[i]}")

#for i in tqdm(range(len(prompt_paths))):
#    prompt_path = prompt_paths[i]
#    scores = []
#
#    for j in range(16):
#        sequence = sequences[i * 16 + j]
#
#        sequence = 'query: ' + str(sequence)
#        text = 'passage: ' + str(read_text_file(prompt_path))
#
#        input_texts = [sequence, sequence, text, text]
#        batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
#        output = model(**batch_dict)
#        embeddings = average_pool(output.last_hidden_state, batch_dict['attention_mask'])
#        embeddings = F.normalize(embeddings, p=2, dim=1)
#        output = (embeddings[:2] @ embeddings[2:].T)
#
#        output = output[0][0]
#        output = output.detach().numpy()
#       scores.append(output)
#    #outputs.append(sum(scores) / len(scores))
#    #outputs.append(max(scores))
#    outputs.append(min(scores))
#
#for i in range(len(prompt_paths)):
#    print(f"{i + 1}:")
#    for j in range(16):
#        print(sequences[i * 16 + j])
#    print(f"score:{outputs[i]}")

print("comparing finished.")

csv = pd.read_csv("./Ground_Truth.csv")
ground_score = []

for i in range(len(prompt_paths)):
    index = 'q' + str(3 * i + 2)
    ground_score.append(csv[index].mean())

ground_score = min_max_normalization(ground_score)

print(f"Final average score: {sum(outputs)/len(outputs)}, Total videos: {len(outputs)}")

S_scores = pd.Series(outputs)
S_ground_score = pd.Series(ground_score)

print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
print(f"Kendall's  : {S_scores.corr(S_ground_score, method='kendall')}")
