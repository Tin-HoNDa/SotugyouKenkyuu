import os
import torch
import cv2
from transformers import CLIPModel, AutoTokenizer
import time
import pandas as pd
# import wandb
from tqdm import tqdm

def min_max_normalization(l):
    #l_min = min(l)
    #l_max = max(l)
    l_min = 1
    l_max = 5
    return [(i - l_min) / (l_max - l_min) for i in l]

def calculate_clip_score(video_path, text, model, tokenizer):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Extract frames from the video
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = [torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames]

    # Initialize an empty tensor to store the concatenated features
    concatenated_features = torch.tensor([], device=device)

    # Generate embeddings for each frame and concatenate the features
    with torch.no_grad():
        for frame in tensor_frames:
            frame_input = frame.unsqueeze(0).to(device)  # Add batch dimension and move the frame to the device
            frame_features = model.get_image_features(frame_input)
            concatenated_features = torch.cat((concatenated_features, frame_features), dim=0)

    # Tokenize the text
    text_tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Convert the tokenized text to a tensor and move it to the device
    text_input = text_tokens["input_ids"].to(device)

    # Generate text embeddings
    with torch.no_grad():
        text_features = model.get_text_features(text_input)

    # Calculate the cosine similarity scores
    concatenated_features = concatenated_features / concatenated_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    clip_score_frames = concatenated_features @ text_features.T

    # Calculate the average CLIP score across all frames, reflects temporal consistency
    clip_score_frames_avg = clip_score_frames.mean().item()

    return clip_score_frames_avg

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()

if __name__ == '__main__':
    dir_videos = "./アンケート/videos/"
    dir_prompts =  "./アンケート/prompts/"

    csv = pd.read_csv("./Ground_Truth.csv")

    video_paths = [os.path.join(dir_videos, x) for x in os.listdir(dir_videos)]
    prompt_paths = [os.path.join(dir_prompts, os.path.splitext(os.path.basename(x))[0]+'.txt') for x in video_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained("./EvalCrafter/Evalcrafter/checkpoints/clip-vit-base-patch32").to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained("./EvalCrafter/Evalcrafter/checkpoints/clip-vit-base-patch32")

    scores = []

    test_num = len(video_paths)
    for i in tqdm(range(len(video_paths))):
        count = 0
        video_path = video_paths[i]
        prompt_path = prompt_paths[i]
        if count == test_num:
            break
        else:
            
            text = read_text_file(prompt_path)
            # ipdb.set_trace()
            score = calculate_clip_score(video_path, text, clip_model, clip_tokenizer)
            count+=1
            # logging.info(f"Vid: {os.path.basename(video_path)}, Pro: {os.path.basename(prompt_path)}, Current clip_score: {score}, Current max clip_score: {m_score}")

        scores.append(score)
        average_score = sum(scores) / len(scores)
        #logging.info(f"Vid: {os.path.basename(video_path)},  Current clip_score: {score}, Current avg. clip_score: {average_score}")

    ground_score = []

    for i in range(len(video_paths)):
        index = 'q' + str(3 * i + 2)
        ground_score.append(csv[index].mean())

    ground_score = min_max_normalization(ground_score)

    for i in range(len(video_paths)):
        print(f"{i+1}, Clip Score:{scores[i]}, ground = {ground_score[i]}")

    # Calculate the average SD score across all video-text pairs
    print(f"Final average clip_score: {average_score}, Total videos: {len(scores)}")

    S_scores = pd.Series(scores)
    S_ground_score = pd.Series(ground_score)

    print(f"Spearman's : {S_scores.corr(S_ground_score, method='spearman')}")
    print(f"Kendall's : {S_scores.corr(S_ground_score, method='kendall')}")
