import os, sys
from glob import glob
import subprocess
import yt_dlp
import librosa
import numpy as np
import cv2
import os
from insightface.app import FaceAnalysis

FPS = 25

def mov_to_mp4():
    files = glob('*.MOV')

    for i in files:
        fname = i.split('.')[0]
        out = fname + '.mp4'
        os.system('ffmpeg -y -i {} -c copy {}'.format(i, out))
        os.remove(i)




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

def extract_audio(vid_path):
    file_name = os.path.basename(vid_path)
    abs_path = os.path.dirname(vid_path)
    out_filename  = file_name.replace('.mp4', '.wav')
    out_path = os.path.join(abs_path, out_filename )

    if os.path.exists(out_path):
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

def get_frame_audio_pairs(video_path, audio_path):
    # Load mel spectrogram
    mel = load_audio_mel(audio_path)

    # Prepare face detector
    face_detector = FaceAnalysis(name='buffalo_l')
    face_detector.prepare(ctx_id=0)

    # Read video frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0, "Could not determine video FPS"

    frames = []
    mels = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % int(fps / FPS) != 0:
            frame_idx += 1
            continue  # downsample to 25 fps

        faces = face_detector.get(frame)
        if not faces:
            frame_idx += 1
            continue

        # Crop and resize face
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            frame_idx += 1
            continue
        face = cv2.resize(face, (96, 96))

        # Align mel index (assuming 80 mel frames/sec, 25 FPS video)
        mel_idx = frame_idx * 4
        if mel_idx + 16 > len(mel):
            break

        mel_chunk = mel[mel_idx : mel_idx + 16]

        frames.append(face)
        mels.append(mel_chunk)
        frame_idx += 1

    cap.release()
    return frames, mels

    


if __name__ == "__main__":

    videos_path = '/home/nikit/benchmarking-videos'

    youtube_list = {
        "hindi_pm_modi_1": "https://www.youtube.com/watch?v=hY-sBLhEpbw",
        "hindi_sujeet_govindani_1":"https://youtu.be/DRWUYp1ReA8",
        "hindi_suresh_srinivasan_1":"https://youtu.be/MUelRUNF180",
        "english_sundar_pichai_1":"https://youtu.be/ic5O2sxhH9M",
        "english_palki_sharma_1":"https://www.youtube.com/watch?v=JwHCOI8V4LQ",
        "hindi_akshata_deshpande_1":"https://www.youtube.com/watch?v=dN9tmgBGNeo",
        "english_anand_mahindra_1":"https://www.youtube.com/watch?v=HF8w15f1dj4",
        
        "english_sjaishankar_1":"https://www.youtube.com/watch?v=-6qN3bm2RYo", #00:01:00 - 00:10:00 - usefull,
        "hindi_vanika_sangtani_1":"https://www.youtube.com/watch?v=JYuWWVdhil4", #00:02:00 - 00:06:00 - usefull
        "english_rashmika_mandana_1":"https://www.youtube.com/watch?v=EQYWOQ-12c8"
 }

    videos = os.listdir(videos_path)
    print(videos)
    for title, url in youtube_list.items():
        if not any(title in _ for _ in videos):
            try:
                download_video(url, title=title)
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
        samples = 5
        videos = os.listdir(os.path.join(videos_path, vid_dir))
        for vid in videos:
            if samples == 0:
                break
            if vid.endswith('.mp4'):
                path = os.path.join(videos_path, vid_dir, vid)
                extract_audio(path)
                samples -= 1




    
