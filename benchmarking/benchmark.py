import benchmarking_utils as utils
from identity_eval import compare_identity as identity_score
from identity_eval import *
from syncnet_eval import main as audio_lip_sync_score
import human_feedback as human_score
import numpy as np
import sys
import os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from os.path import isdir, dirname, basename, exists, join
import glob
import argparse
import yaml
import csv
import warnings
import subprocess
import pandas as pd

warnings.filterwarnings('ignore')


class Benchmark:
    def __init__(self, source_path, args):
        self.source_path = source_path
        self.inference_args = args

    def prepare_video_audio_pairs(self):
        #source path 
        #inference config
        self.video_audio_pairs = []
        self.ground_truth = []
        
        alread_passed_pairs = []

        def make_video_audio_pairs(d1, d2):
            videos = glob.glob(join(d1, '*'))
            audios = glob.glob(join(d2, '*'))
            video_path = np.random.choice(list(filter(lambda x: x.endswith('.mp4'), videos)))
            audio_path = np.random.choice(list(filter(lambda x: x.endswith('.wav'), audios)))
            
            self.ground_truth.append(audio_path.replace('.wav', '.mp4'))
            self.video_audio_pairs.append([str(video_path), str(audio_path)])
        
        dirs = glob.glob(join(self.source_path, '*'))
        
        print(f'Video directories at {self.source_path}: {dirs}')
        dir_pairs = []

        for d in dirs:
            x = [[d, i] for i in dirs if i != d and basename(i).split('_')[1] == basename(d).split('_')[1]]
            dir_pairs.extend(x)
        
        
        for d1, d2 in dir_pairs:
            if [d1, d2] in alread_passed_pairs:
                continue
            make_video_audio_pairs(d1, d2)
            alread_passed_pairs.append([d1, d2])


        c = {
            f"task_{i}":{
                "video_path": x[0],
                "audio_path":x[1],
                "result_name": 'video_{}_audio_{}'.format(basename(x[0]).replace('.mp4',''), basename(x[1]).replace('.wav', '.mp4'))
            } for i, x in enumerate(self.video_audio_pairs)
        }
        
        with open(self.inference_args.inference_config, 'w') as file:
            yaml.dump(c, file, default_flow_style=False, sort_keys=False)
        print(f'Finish writing task to yaml file: {json.dumps(c, indent=4)}')
        return self.video_audio_pairs
    
    def prepare_video_with_same_audio(self, voice_list: dict):
        dirs = os.listdir(self.source_path)

        pairs = []

        def func(vd):
            videos = glob.glob(join(vd, '*.mp4'))
            video_path = np.random.choice(list(filter(lambda x: x.endswith('.mp4'), videos)))
            return video_path

        for d in dirs:
            gender = d.split('_')[1]
            if gender == 'male':
                x = voice_list.get('male')
            elif gender == 'female':
                x = voice_list.get('female')
            
            vd = join(self.source_path, d)
            vid = str(func(vd))
            pairs.append([vid, join(self.source_path, x[0])])
            pairs.append([vid, join(self.source_path, x[1])])
    
        c = {
            f"task_{i}":{
                "video_path": x[0],
                "audio_path":x[1],
                "result_name": 'video_{}_audio_{}'.format(basename(x[0]).replace('.mp4',''), basename(x[1]).replace('.wav', '.mp4'))
            } for i, x in enumerate(pairs)
        }
        with open(self.inference_args.inference_config, 'w') as file:
            yaml.dump(c, file, default_flow_style=False, sort_keys=False)
        print(f'Finish writing task to yaml file: {json.dumps(c, indent=4)}')
                    
    
    def benchmarking(self, ref_gen_video_pairs, output_csv_path = None, model_stage = 'original' ):

        csv_path = output_csv_path or  join(dirname(ref_gen_video_pairs[0][1]), f'{model_stage}_benchmarking.csv')
    
        
        score_data = {
            "ground_truth_video_name": [],
            "generate_video_name": [],
            "lip_sync_score":[],
            "cosine_similarity":[],
            "psnr":[],
            "ssim":[],
            "lpips":[],
            "fid":[],
            "compare_landmarks":[],
            "compare_emotions":[],
            "flicker_score_ground_truth":[],
            "flicker_score_gen_video":[],
            "ptical_flow_consistency_ground_truth":[],
            "ptical_flow_consistency_gen_video":[]
            
        }

        for ref_video, gen_video in ref_gen_video_pairs:
            try:
                score_data['ground_truth_video_name'].append(basename(ref_video))
                score_data['generate_video_name'].append(basename(gen_video))
                score_data['lip_sync_score'].append(0)#append(audio_lip_sync_score(ref_video.replace('.mp4','.wav'), gen_video, self.inference_args))
                score_data['cosine_similarity'].append(identity_score(ref_video, gen_video))
                score_data['compare_emotions'].append(compare_emotions(ref_video, gen_video))

                ref_frames_dir = extract_video_frames(ref_video)
                gen_frames_dir = extract_video_frames(gen_video)

                score_data['fid'].append(compute_fid(ref_frames_dir, gen_frames_dir))
                score_data['flicker_score_ground_truth'].append(compute_optical_flow_consistency(ref_frames_dir))
                score_data['flicker_score_gen_video'].append(compute_optical_flow_consistency(gen_frames_dir))
                score_data['ptical_flow_consistency_ground_truth'].append(compute_optical_flow_consistency(ref_frames_dir))
                score_data['ptical_flow_consistency_gen_video'].append(compute_optical_flow_consistency(gen_video))

                scores = average_score(ref_frames_dir, gen_frames_dir)
                for key, value in scores.items():
                    score_data[key].append(np.average(np.array(value)))
                    print(f'Average score for {key}:  {score_data[key]}')
            except Exception as e:
                print(f'Got an error while calculating {ref_video} and {gen_video} score...')
                print(e)
                continue
        
        print('score_data', score_data)
        try:
            df = pd.DataFrame(score_data)
            print(f'Sample rows: {df.head(4)}')
            df.to_csv(csv_path)
            print(f'Benchmarking done on {model_stage} model')
        except Exception as e:
            print(f'Error in storing panda DF: {e}')
            with open(csv_path.replace('.csv', '.json'), 'w') as f:
                json.dump(score_data, f, indent=4)
                print(f"created a json file at {csv_path.replace('.csv', '.json')}")

def return_ref_video_paths(model_results_path, source_path = './benchmarking_videos'):
    dirs = os.listdir(model_results_path)
    #video_clip002_english_anand_mahindra_1_audio_clip004_english_sundar_pichai_1.mp4
    ref_gen_video_pairs = []
    for v in dirs:
        if v.endswith('.mp4'):
            ref_vid_name  = v.split('_audio_')[1]
            ref_dir_name = '_'.join(ref_vid_name.split('_')[2:]).split('.')[0]
            print(f'Video reference dir name {ref_dir_name}')
            for dd in os.listdir(source_path):
                if ref_dir_name in dd:
                    ref_dir_name = dd
            ref_gen_video_pairs.append([
                join(source_path, ref_dir_name, ref_vid_name),
                join(model_results_path, v)
            ])
    
    print(f'Reference-generate video pairs : {ref_gen_video_pairs}')
    return ref_gen_video_pairs


def main(args):
    original_mode = 'original'
    finetuned_mode = 'finetuned'
    
    benchmark = Benchmark(args.source_path, args)

    original_model_results = join(args.result_dir, original_mode, args.version)
    output_csv_path = join(original_model_results, f'{original_mode}_benchmarking.csv')
    ref_gen_video_pairs = return_ref_video_paths(original_model_results, args.source_path)

    print(f'Starting benchmarking onf {original_mode}')
    benchmark.benchmarking(ref_gen_video_pairs, output_csv_path = output_csv_path, model_stage = original_mode)
    
    finetuned_model_results = join(args.result_dir, finetuned_mode, args.version)
    output_csv_path = join(finetuned_model_results, f'{finetuned_mode}_benchmarking.csv')
    ref_gen_video_pairs = return_ref_video_paths(finetuned_model_results, args.source_path)

    print(f'Starting benchmarking onf {finetuned_mode}')
    benchmark.benchmarking(ref_gen_video_pairs, output_csv_path = output_csv_path, model_stage = finetuned_mode)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml", help="Path to inference configuration file")
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")
   

    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json", help="Path to UNet configuration file")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth", help="Path to UNet model weights")
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")
    parser.add_argument("--stage", type=str, default='original', help="Stage to select for getting output from model.")
    parser.add_argument("--source_path", type = str, default='./benchmarking_videos')
    parser.add_argument("--force_generate", type=bool, default=False, help="Force generate video even if already exists.")


    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for faster inference")
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Model version to use")
    args = parser.parse_args()
    main(args)

