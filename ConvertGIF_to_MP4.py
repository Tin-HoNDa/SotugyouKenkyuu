import moviepy.editor as mp
from pathlib import Path
import os

for file in os.listdir("アンケート/videos/"):
        print(file)
        name, extension = file.split(".")
        if extension == "gif":
            IMAGE_PATH = "アンケート/videos/" + file
            movie_file=mp.VideoFileClip(IMAGE_PATH)
            dir_dest = Path("mp4/")
            if not dir_dest.is_dir():
                dir_dest.mkdir(0o700)
            movie_file.write_videofile("mp4/" + name + ".mp4")
            movie_file.close()
