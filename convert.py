import os, sys
from glob import glob


files = glob('*.MOV')

for i in files:
    fname = i.split('.')[0]
    out = fname + '.mp4'
    os.system('ffmpeg -y -i {} -c copy {}'.format(i, out))

    #os.system('ffmpeg -y -i {} -c:v libx264 -preset veryslow -crf 0 -c:a copy {}'.format(i, out))

