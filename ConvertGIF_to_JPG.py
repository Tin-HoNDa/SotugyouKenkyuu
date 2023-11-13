from PIL import Image, ImageSequence
from pathlib import Path
import os

def get_frames(path):
    '''パスで指定されたファイルのフレーム一覧を取得する
    '''
    im = Image.open(path)
    return (frame.copy() for frame in ImageSequence.Iterator(im))

def write_frames(frames, name_original, destination, model):
    '''フレームを別個の画像ファイルとして保存する
    '''
    path = Path(name_original)

    stem = path.stem
    extension = path.suffix

    # 出力先のディレクトリが存在しなければ作成しておく
    dir_dest = Path(destination + "/" + model)
    print(dir_dest)
    if not dir_dest.is_dir():
        dir_dest.mkdir(0o700)
        print('Destionation directory is created: "{}".'.format(destination))

    for i, f in enumerate(frames):
        name = '{}/{}/{}-{}{}'.format(destination, model, stem[0], i + 1, extension)
        f.save(name)
        print('A frame is saved as "{}".'.format(name))

# passes = ["1-ToonYou", "2-Lyriel", "3-RcnzCartoon", "4-MajicMix", "5-RealisticVision", "6-Tusun", "7-FilmVelvia", "8-GhibliBackground"]
passes = ["1-ToonYou", "2-Lyriel", "3-RcnzCartoon", "4-MajicMix", "5-RealisticVision", "7-FilmVelvia", "8-GhibliBackground"]

for models in passes:
    for file in os.listdir("AnimateDiff/AnimateDiff/samples/" + models + "/sample"):
        print(file)
        name, extension = file.split(".")
        if extension == "gif":
            IMAGE_PATH = "AnimateDiff/AnimateDiff/samples/" + models + "/sample/" + file
            frames = get_frames(IMAGE_PATH)
            write_frames(frames, IMAGE_PATH, 'splitted', models)