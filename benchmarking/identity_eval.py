
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import torch

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
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--gen', required=True)
    args = parser.parse_args()

    score = compare_identity(args.ref, args.gen)
    print(f'Cosine Similarity Score (ArcFace): {score:.4f}')
