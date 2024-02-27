# Text-to-Videoモデルを用いて生成された動画に対する定量的評価指標の検討とその評価

ここは私の卒業研究用のリポジトリです．公開することを想定していなかったため大分ぐちゃぐちゃです．

## 実験手順

### 動画像生成

[AnimateDiffのリポジトリ](https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md) に従い導入してください．プロンプトは[EvalCrafterのもの](https://github.com/evalcrafter/EvalCrafter/tree/master/prompts) を使用しました．

### スコア計測

```yaml
Python=3.10
torch=1.13.1
transformers=4.30.2
```
後は何でもいいと思う

言語埋込モデルの変更などはコメントアウトをいい感じにつけたり外したりしてください．

[TimeSformer-GPT2](https://huggingface.co/Neleac/timesformer-gpt2-video-captioning) を使用する場合：

```bash
python "V2S_TimeSformer-GPT2%20Video%20Captioning.py"
```

[Vision Transformer+GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) を使用する場合：

```bash
python V2S_vit-gpt2-image-captioning.py
```

CLIPScore を使う場合：

```bash
python Clip_Score.py
```

### 可視化など

`caption.ipynb` で行っていましたが，普通にぐちゃぐちゃなので新しく作ったほうが早いと思います．

## 作成した資料など

発表資料などは `卒論` フォルダに，論文の翻訳は `工業英語` フォルダに入っています．活用しても構いませんが，工業英語の課題は自力でやりましょう．思っているより早く終わると思います．

## その他

フォルダの中身は逐一整理しましょう．
