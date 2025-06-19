import benchmarking_utils as utils
from identity_eval import compare_identity as identity_score
from syncnet_eval import main as audio_lip_sync_score
import human_feedback as human_score
import numpy as np
import sys
import os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.inference import main as generate_animation
from os.path import isdir, dirname, basename, exists, join
import glob
import argparse
import yaml
import csv
import warnings
import subprocess

warnings.filterwarnings('ignore')


class Benchmark:
    def __init__(self, source_path, args):
        self.source_path = source_path
        self.inference_args = args
        self.prepare_video_audio_pairs()

    def prepare_video_audio_pairs(self):
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
    

    def inferenace(self):
        # self.inference_config = args.inference_config
        # self.result_dir = args.result_dir
        # self.unet_model_path = args.unet_model_path
        # self.unet_config = args.unet_config
        # self.version_arg = args.version_arg
        #syncnet checkpoint
        #syncnet config
        if self.inference_args.stage == 'finetuned':
            print('Starting inference on benchmarking data. Finetuned model')
            self.inference_args.result_dir = self.inference_args.result_dir_finetuned 
            generate_animation(self.inference_args)
            self.benchmarking(model_stage = 'finetuned')
        else:
            print('Starting inference on benchmarking data. Original Model')
            self.inference_args.result_dir = self.inference_args.result_dir_original
            generate_animation(self.inference_args)
            self.benchmarking(model_stage='original')
                    
    
    def benchmarking(self, model_stage = 'original' ):
        results_path = join(self.inference_args.result_dir, self.inference_args.version)
        self.generated_videos = glob.glob(join(results_path, '*.mp4'))
        
        headers = ['index', 'video_name', 'identity_score', 'sync_score', 'human_feedback']
        with open(join(results_path, f'{model_stage}_benchmarking.csv'), 'w+') as csv_out:
            csv_object = csv.writer(csv_out)
            csv_object.writerow(headers)
            
            i = 1
            for ref, gen in zip(self.ground_truth, self.generated_videos):
                row = [i, basename(self.ground_truth[i-1])]
                print(f'Calculating identity score for {ref} with generated video {gen}')
                row.append(identity_score(ref, gen))

                print(f"Calculating lip sync score for generated {gen} with audio {ref.replace('.mp4', '.wav')}")
                row.append(audio_lip_sync_score(gen , ref.replace('.mp4', '.wav'), self.inference_args))
                row.append(0)

                csv_object.writerow(row)
                i+=1
        print(f'Benchmarking done on {model_stage} model')


def main(args):
    benchmark = Benchmark(args.source_path, args)
    benchmark.inferenace()

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
    parser.add_argument("--result_dir_finetuned", default='./results_finetuned', help="Directory for output results")
    parser.add_argument("--result_dir_original", default='./results_original', help="Directory for output results")
    parser.add_argument("--stage", type=str, default='original', help="Stage to select for getting output from model.")
    parser.add_argument("--source_path", required=True)
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

