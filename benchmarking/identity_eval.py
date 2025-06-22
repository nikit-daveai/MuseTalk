
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from torchvision import transforms
from scipy.spatial.distance import cosine
from scipy.spatial import procrustes

import os
from benchmarking_utils import *
# from deepface import DeepFace  # uses a pre-trained emotion classifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Note: real_frames, and generated_frames are global variables containing np.array of each lip extracted from video

# Visual Quality Metrics
def average_score(ref_frames_dir, gen_frames_dir):
    global real_frames,  generated_frames, ref_coords, gen_coords

    scores = {
        "psnr":[],
        "ssim":[],
        "lpips":[],
        "euclidean_distance":[],
        "procrustes_distance":[],
        "compare_landmarks":[]
    }
    ref_frames_dir, ref_flatten_coords = get_frames_from_dir(ref_frames_dir)
    gen_frames_dir, gen_flatten_coords = get_frames_from_dir(gen_frames_dir)
    
    #setting global values
    real_frames = copy(ref_frames_dir)
    generated_frames = copy(gen_frames_dir)
    ref_coords = copy(ref_flatten_coords)
    gen_coords = copy(gen_flatten_coords)
 
    for img1, img2 in zip(ref_frames_dir, gen_frames_dir):
        scores['psnr'].append(compute_psnr(img1, img2))
        scores['ssim'].append(compute_ssim(img1, img2))
        scores['lpips'].append(compute_lpips(img1, img2))

    for lmark1, lmark2 in zip(ref_flatten_coords, gen_flatten_coords):
        scores['euclidean_distance'].append(compute_euclidean_distance(lmark1, lmark2))
        scores['procrustes_distance'].append(compute_procrustes_distance(lmark1, lmark2))
        scores['compare_landmarks'].append(compare_landmarks(lmark1, lmark2))
    
    return scores


def image_to_tensor(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(img).unsqueeze(0)


def compute_euclidean_distance(lips1, lips2):
    '''
    #measure euclidean distance between real and generated lips
    #Raw distance between each corresponding landmark point
    #pixel-level differences when everything is aligned
    '''

    dists = np.linalg.norm(lips1 - lips2, axis=1)
    dists = np.mean(dists)
    print(f'Euclidean distance: {dists}')
    return dists

def compute_procrustes_distance(lips1, lips2):
    '''
    #normalized shapes and the dissimilarity score
    #measures shape similarity invariant to scale, rotation, and translation.
    '''
    
    mtx1, mtx2, disparity = procrustes(lips1, lips2)
    print(f'Procrustes_distance score : {disparity}')
    return disparity


def compare_identity():
    '''
    # comparison, rough shape similarity, but can be mis leading with noise.
    '''
    def get_embedding(lips_frames):
        stacked_lips_frames = np.stack(lips_frames, axis = 0) #N H W C #adds new axis for number of frames at 0
        return np.mean(stacked_lips_frames, axis=0)

    emb1 = get_embedding(ref_coords).flatten()
    emb2 = get_embedding(gen_coords).flatten()

    cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f'Cosine Similarity: {cosine_similarity}')
    return cosine_similarity


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    score = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    print(f'PNSR Score : {score}')
    return score

def compute_ssim(img1, img2):
    if min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1]) < 7:
        print("Image too small for SSIM. Returning 0.")
        return 0.0
    
    try:
        score = ssim(img1, img2, channel_axis=-1)  # Use channel_axis instead of multichannel=True
        print(f'SSIM Score : {score}')
        return score
    except Exception as e:
        print(f"SSIM computation failed: {e}")
        return 0.0

def compute_lpips(img1, img2):
    img1_tensor = image_to_tensor(img1)
    img2_tensor = image_to_tensor(img2)
    loss_fn = lpips.LPIPS(net='alex')
    score = loss_fn(img1_tensor, img2_tensor).item()
    print(f'LPIPS Score {score}')
    return score

def compute_fid(ref_frames_dir, fake_frames_dir):
    score = fid_score.calculate_fid_given_paths([ref_frames_dir, fake_frames_dir], batch_size=8, device=device, dims=2048)
    print(f'FID Score: {score}')
    return score

'''
#  Expression & Emotion Preservation
def predict_emotions(frames):
    predictions = []
    for frame in frames:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            predictions.append(emotion)
        except Exception as e:
            predictions.append("unknown")
    return predictions

#not needed
def compare_emotions(ref_video, gen_video):
    real_frames = ref_video[:50]
    fake_frames = gen_video[:50]
 
    pred_real = predict_emotions(real_frames)
    pred_fake = predict_emotions(fake_frames)

    valid_pairs = [(r, f) for r, f in zip(pred_real, pred_fake) if r != "unknown" and f != "unknown"]
    if not valid_pairs:
        return 0.0

    matches = sum([r == f for r, f in valid_pairs])
    score = matches / len(valid_pairs)
    print(f'Compare emotions score : {score}')
    return score
'''

#Temporal Consistency
def compute_flicker_score(tag):
    
    if tag == 'real':
        frames = real_frames
    else:
        frames = generated_frames

    diffs = [np.mean((frames[i] - frames[i+1]) ** 2) for i in range(len(frames) - 1)]
    score = np.mean(diffs)
    print(f'Flicker Score: {score}')
    return score

def compute_optical_flow_consistency(tag):
    if tag == 'real':
        frames = real_frames
    else:
        frames = generated_frames
    
    flows = []
    target_size = (frames[0].shape[1], frames[0].shape[0])  # width, height
    frames = [cv2.resize(f, target_size) for f in frames] #resize based on first frame
 
    prev = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev = curr
    magnitudes = [np.mean(np.linalg.norm(f, axis=2)) for f in flows]
    score = np.std(magnitudes)
    print(f'Optical Flow score: {score}')
    return score

# Pose & Landmark Accuracy
#not very usefule

def compare_landmarks(landmarks_real, landmarks_fake):
    if landmarks_real is None or landmarks_fake is None:
        return None
    score = np.mean(np.linalg.norm(landmarks_real - landmarks_fake, axis=1))
    print(f'Compare landmark: {score}')
    return score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--gen', required=True)
    args = parser.parse_args()

    score = compare_identity(args.ref, args.gen)
    print(f'Cosine Similarity Score (ArcFace): {score:.4f}')
