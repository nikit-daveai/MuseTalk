
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import torch
import numpy as np
import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from pytorch_fid import fid_score
from torchvision import transforms
from scipy.spatial.distance import cosine
import mediapipe as mp
import os
from deepface import DeepFace  # uses a pre-trained emotion classifier

from benchmarking_utils import extract_video_frames

def compare_identity(video_path_1, video_path_2):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0)

    def get_embedding(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = app.get(frame)
            if faces:
                emb = faces[0].embedding
                frames.append(emb)
        cap.release()
        return np.mean(frames, axis=0)

    emb1 = get_embedding(video_path_1)
    emb2 = get_embedding(video_path_2)
    cosine_similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cosine_similarity


# Visual Quality Metrics
def average_score(ref_frames_dir, gen_frames_dir):
    scores = {
        "psnr":[],
        "ssim":[],
        "lpips":[],
        "compare_landmarks":[]
    }

    for img1, img2 in zip(ref_frames_dir, gen_frames_dir):
        score['psnr'].append(compute_psnr(img1, img2))
        score['ssim'].append(compute_ssim(img1, img2))
        score['lpips'].append(compute_lpips(img1, img2))
        score['compare_landmarks'].append(compare_landmarks(img1, img2))
    
    return scores


def image_to_tensor(img):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return preprocess(img).unsqueeze(0)

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def compute_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

def compute_lpips(img1, img2):
    img1_tensor = image_to_tensor(img1)
    img2_tensor = image_to_tensor(img2)
    loss_fn = lpips.LPIPS(net='alex')
    return loss_fn(img1_tensor, img2_tensor).item()

def compute_fid(ref_frames_dir, fake_frames_dir):
    return fid_score.calculate_fid_given_paths([ref_frames_dir, fake_frames_dir], batch_size=8, device='cuda', dims=2048)

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

def compare_emotions(ref_video, gen_video):
    real_frames = extract_video_frames(ref_video, max_frames=50)
    fake_frames = extract_video_frames(gen_video, max_frames=50)

    pred_real = predict_emotions(real_frames)
    pred_fake = predict_emotions(fake_frames)

    valid_pairs = [(r, f) for r, f in zip(pred_real, pred_fake) if r != "unknown" and f != "unknown"]
    if not valid_pairs:
        return 0.0

    matches = sum([r == f for r, f in valid_pairs])
    return matches / len(valid_pairs)

#Temporal Consistency


def compute_flicker_score(frames):
    diffs = [np.mean((frames[i] - frames[i+1]) ** 2) for i in range(len(frames) - 1)]
    return np.mean(diffs)

def compute_optical_flow_consistency(frames):
    flows = []
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        prev = curr
    magnitudes = [np.mean(np.linalg.norm(f, axis=2)) for f in flows]
    return np.std(magnitudes)

# Pose & Landmark Accuracy

mp_face_mesh = mp.solutions.face_mesh


def extract_landmarks(frame):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark])
        return None

def compare_landmarks(img1, img2):
    landmarks_real = extract_landmarks(img1)
    landmarks_fake = extract_landmarks(img2)
    if landmarks_real is None or landmarks_fake is None:
        return None
    return np.mean(np.linalg.norm(landmarks_real - landmarks_fake, axis=1))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--gen', required=True)
    args = parser.parse_args()

    score = compare_identity(args.ref, args.gen)
    print(f'Cosine Similarity Score (ArcFace): {score:.4f}')
