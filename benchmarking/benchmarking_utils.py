import os, sys
from glob import glob
import subprocess
import yt_dlp
import librosa
import numpy as np
import cv2
import os
import torch
import random
from os.path import isdir, dirname, basename, exists, join
from PIL import Image
import mediapipe as mp
from copy import deepcopy as copy
#from syncnet_eval import generated_frames, gen_coords



FPS = 25

def mov_to_mp4():
    files = glob('*.MOV')

    for i in files:
        fname = i.split('.')[0]
        out = fname + '.mp4'
        os.system('ffmpeg -y -i {} -c copy {}'.format(i, out))
        os.remove(i)


def crop_to_common_size(img1, img2):
    '''
    im1, im2 are numpy arrays
    '''
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1_cropped = img1[:h, :w]
    img2_cropped = img2[:h, :w]
    return img1_cropped, img2_cropped

def pad_and_resize(img, target_size=(128, 128)):
    h, w = img.shape[:2]
    
    # scaling factor and resize while preserving aspect ratio
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # padding values
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded


def extract_mouth_from_frames_padded(frame_paths, mouth_padding=10, output_size=(128, 128)):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    lips_frames = []
    flatten_coords =[]
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                lip_indices = list(range(61, 89)) + list(range(89, 96))
                xs = [int(face_landmarks.landmark[i].x * w) for i in lip_indices]
                ys = [int(face_landmarks.landmark[i].y * h) for i in lip_indices]
                
                coords = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in lip_indices])

                x_min = max(min(xs) - mouth_padding, 0)
                y_min = max(min(ys) - mouth_padding, 0)
                x_max = min(max(xs) + mouth_padding, w)
                y_max = min(max(ys) + mouth_padding, h)

                lips_crop = frame[y_min:y_max, x_min:x_max]


                #padded_crop = pad_and_resize(lips_crop, target_size=output_size)
                padded_crop = cv2.resize(lips_crop, output_size)

                lips_frames.append(padded_crop)
                flatten_coords.append(coords)
                cv2.imwrite(fp, padded_crop)

    return lips_frames, flatten_coords

def get_frames_from_dir(frames_dir):
    frame_paths = glob(join(frames_dir, '*.png'))
    return extract_mouth_from_frames_padded(frame_paths)
    
    frames = []
    for fp in frame_paths:
        frames.append(
            np.array(Image.open(fp).convert('RGB'))
        )

    print(f'Frame shape from get_frames_from_dir: {frames[0].shape}')
    return frames


def video_to_25fps(vid_path):
    file_name = os.path.basename(vid_path)
    abs_path = os.path.dirname(vid_path)
    out_filename  = 'final_{}'.format(file_name)
    out_path = os.path.join(abs_path, out_filename )
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-y", "-i", vid_path, 
            "-r", "25", "-crf", "15", "-c:v", "libx264", 
            "-pix_fmt", "yuv420p", '{}'.format(out_path)
            ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting video to 25FPS {vid_path}: {e}")
    
    os.remove(vid_path)
    os.rename(out_path, vid_path)


def split_video(vid_path, duration = 30):
    original_filename = os.path.basename(vid_path)
    
    out_path = os.path.join(os.path.dirname(vid_path), original_filename.replace('.mp4', ''))
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    command = [
                'ffmpeg', '-i', vid_path, '-c', 'copy', '-map', '0',
                '-segment_time', str(duration), '-f', 'segment',
                '-reset_timestamps', '1',
                os.path.join(out_path, f'clip%03d_{original_filename}')
            ]

    subprocess.run(command, check=True)
    os.remove(vid_path)

def extract_audio(vid_path, force=False):
    file_name = os.path.basename(vid_path)
    abs_path = os.path.dirname(vid_path)
    out_filename  = file_name.replace('.mp4', '.wav')
    out_path = os.path.join(abs_path, out_filename )

    if os.path.exists(out_path) and not force:
        print(f'Files exists at: {out_path}')
        return

    try:
        command = [
            'ffmpeg', '-hide_banner', '-y', '-i', vid_path,
            '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
            '-ar', '16000', '-ac', '1', out_path,
        ]
        
        subprocess.run(command, check=True) 
        print(f"Audio saved to: {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {vid_path}: {e}")

def to_list(x):
    if not isinstance(x, list):
        return [x]
    return x

def download_video(vid_url, timestamp_ranges = '00:00:10-00.06:10', dest_path = '/home/nikit/benchmarking-videos', title= None):
    timestamp_ranges = to_list(timestamp_ranges)
    try:
        ydl_opts = {
        'format': 'bestvideo+bestaudio',
        'outtmpl': os.path.join(dest_path, title if title else '%(title)s'),
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'  # forces re-encoding
        }]}
        if timestamp_ranges:
            ydl_opts['download_sections'] = {'*': timestamp_ranges}
    except Exception as e:
        print(e)
        print('retrying with without timestamp....')
        #download_video(vid_url , timestamp_ranges= None)
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([vid_url])


def load_audio_mel(audio_path, sr=16000, hop_length=160, win_length=400, n_mels=80):
    y, _ = librosa.load(audio_path, sr=sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=512, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max).T  # shape: (T, n_mels)
    return mel


def extract_video_frames(video_path, max_frames=None, every_nth=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_nth == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        count += 1
    cap.release()
    return frames


def get_frame_audio_pairs(generated_frames, audio_path, num_frames=16, target_fps=25):
    """
    Extracts face frame stacks and matching mel-spectrogram chunks.
    Returns tensors ready for model input.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mel = load_audio_mel(audio_path)  # still a NumPy array for now

    #modify when computing lip sync score
    all_faces =  generated_frames

    if len(all_faces) < num_frames:
        print(f"[WARN] Not enough frames ({len(all_faces)}). Skipping.")
        return [], []

    frames = []
    mels = []

    for i in range(len(all_faces) - num_frames + 1):
        frame_window = all_faces[i:i + num_frames]

        # Stack and convert to PyTorch tensor: [num_frames, 3, 96, 96]
        stacked = torch.from_numpy(np.stack(frame_window)).permute(0, 3, 1, 2).float() / 255.0
        stacked = stacked.reshape(-1, 96, 96)  # [48, 96, 96]
        frame_tensor = stacked.unsqueeze(0).to(device)  # [1, 48, 96, 96]
        frames.append(frame_tensor)

        # Prepare mel tensor
        mel_idx = i * 4
        if mel_idx + 16 > len(mel):
            break
        mel_chunk = mel[mel_idx : mel_idx + 16]
        mel_tensor = torch.from_numpy(mel_chunk).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, 16, 80]
        mels.append(mel_tensor)

    print(f"[INFO] Extracted {len(frames)} frame stacks and {len(mels)} mel chunks")
    return frames, mels


if __name__ == "__main__":

    videos_path = '/MuseTalk/benchmarking_videos'

    youtube_list = {
        # "hindi_pm_modi_1": "https://www.youtube.com/watch?v=hY-sBLhEpbw",
        # "hindi_sujeet_govindani_1":"https://youtu.be/DRWUYp1ReA8",
        # "hindi_suresh_srinivasan_1":"https://youtu.be/MUelRUNF180",
        # "english_sundar_pichai_1":"https://youtu.be/ic5O2sxhH9M",
        # "english_palki_sharma_1":"https://www.youtube.com/watch?v=JwHCOI8V4LQ",
        # "hindi_akshata_deshpande_1":"https://www.youtube.com/watch?v=dN9tmgBGNeo",
        # "english_anand_mahindra_1":"https://www.youtube.com/watch?v=HF8w15f1dj4",
        
        # "english_sjaishankar_1":"https://www.youtube.com/watch?v=-6qN3bm2RYo", #00:01:00 - 00:10:00 - usefull,
        # "hindi_vanika_sangtani_1":"https://www.youtube.com/watch?v=JYuWWVdhil4", #00:02:00 - 00:06:00 - usefull
        # "english_rashmika_mandana_1":"https://www.youtube.com/watch?v=EQYWOQ-12c8"
 }
    youtube_list = {
        "hindi_female_apna_college_1":"https://www.youtube.com/shorts/u3FtKYm6ykE?si=9zURQCSPcQOSNyx4",
        "english_male_manish_1":"https://www.youtube.com/shorts/3JSY2YOM8GY",
        "english_female_aleena_1":"https://www.youtube.com/watch?v=cuwRniQ1-Y8",
        "hindi_female_amrita_1":"https://www.youtube.com/watch?v=Fn1GQk_x1uA",
        "hindi_female_sanchita_1":"https://www.youtube.com/watch?v=GIernkgvQiU",
        "english_male_drpal_1":"https://www.youtube.com/shorts/BA8TJP1s-Qw",
        "english_female_bipasha_1":"https://www.youtube.com/watch?v=d9ZwezO5oRU",
        "hindi_male_sameer_madaan_1":"https://www.youtube.com/watch?v=cWoRvHih3k8",
        "hindi_male_sameer_madaan_2":"https://www.youtube.com/watch?v=QiRZpuGJ-1o",
        "hindi_male_gaurav_taneja_1":"https://www.youtube.com/watch?v=zSuDRcu15H0",
        "hindi_female_simmy_goraya_1":"https://www.youtube.com/watch?v=bPAypCh4k0Q",
        "hindi_male_laksh_vaishnav_1":"https://www.youtube.com/watch?v=67jrSL6RzJM",
        "english_male_savinder_puri_1":"https://www.youtube.com/watch?v=r3YPYdSP-yU",
        "english_male_shubham_gill_1":"https://www.youtube.com/watch?v=behkHHC1kUs",
        "hindi_male_jeet_salal_1":"https://www.youtube.com/watch?v=tddiO_5LNb8",
        "hindi_male_guru_mann_1":"https://www.youtube.com/watch?v=kUErCeb3CcU",
        "english_male_jeet_salal_2":"https://www.youtube.com/watch?v=ngX4TjNHlvU" 
    }

    videos = os.listdir(videos_path)
    dest_path = videos_path
    print(videos)
    for title, url in youtube_list.items():
        if not any(title in _ for _ in videos):
            try:
                download_video(url, title=title, dest_path = dest_path)
            except Exception as e:
                print(f'Failed to download video- {title} : {url}')
                print(e)
                continue
        else:
            print('Video already processed...')

    print(videos)
    
    #video processing and segmentation
    for vid in videos:
        if vid.endswith('.mp4'):
            path = os.path.join(videos_path, vid)
            video_to_25fps(path)
            split_video(path)
 
    #audio extraction
    vid_dirs = os.listdir(videos_path)
    for vid_dir in vid_dirs:
        if not isdir(os.path.join(videos_path, vid_dir)):
            print(f'{vid_dir} is not a directory, skipping...')
            continue
        samples = 10
        videos = os.listdir(os.path.join(videos_path, vid_dir))
        random.shuffle(videos)
        for vid in videos:
            if samples == 0:
                break
            if vid.endswith('.mp4'):
                path = os.path.join(videos_path, vid_dir, vid)
                extract_audio(path, force = True)
                samples -= 1




    
