# AnimateDiff：特別にチューニングを用いないパーソナライズされたテキストから画像への拡散モデルのアニメーション化 <!-- omit in toc -->

## 目次 <!-- omit in toc -->

- [概要](#概要)
- [1 はじめに](#1-はじめに)
- [2 先行研究](#2-先行研究)
  - [Text-to-Image 拡散モデル](#text-to-image-拡散モデル)
  - [パーソナライズ Text-to-Image モデル](#パーソナライズ-text-to-image-モデル)
  - [パーソナライズ T2I アニメーション](#パーソナライズ-t2i-アニメーション)
- [3 手法](#3-手法)
  - [3.1 前置き](#31-前置き)
    - [一般的な T2I 生成器](#一般的な-t2i-生成器)
    - [パーソナライズド画像生成](#パーソナライズド画像生成)
  - [3.2 パーソナライズされたアニメーション](#32-パーソナライズされたアニメーション)
  - [3.3 モーションモデリングモジュール](#33-モーションモデリングモジュール)
    - [ネットワーク拡張](#ネットワーク拡張)
    - [モジュール設計](#モジュール設計)
    - [訓練目標](#訓練目標)
- [4 実験](#4-実験)
  - [4.1 実装の詳細](#41-実装の詳細)
    - [訓練](#訓練)
    - [評価](#評価)
  - [4.2 定性的な結果](#42-定性的な結果)
  - [4.3 ベースラインとの比較](#43-ベースラインとの比較)
  - [4.4 アブレーション研究](#44-アブレーション研究)
- [5 限界と今後の課題](#5-限界と今後の課題)
- [6 おわりに](#6-おわりに)
- [補遺](#補遺)
  - [A 追加の詳細](#a-追加の詳細)
    - [A.1 モデルの多様性](#a1-モデルの多様性)
    - [A.2 定性的な結果](#a2-定性的な結果)

![図1](2023-09-25-10-37-06.png)
図1：AnimateDiffは、パーソナライズされたT2I（text-to-image）モデルを、モデル固有のチューニングなしにアニメーションジェネレータに拡張するための効果的なフレームワークである。AnimateDiffは、大規模な動画データセットからモーションプリオを学習した後、ユーザーによって訓練された、またはCivitAI[^4]やHuggingface[^8]のようなプラットフォームから直接ダウンロードされた、パーソナライズされたT2Iモデルに挿入され、適切なモーションを持つアニメーションクリップを生成することができる。

## 概要

テキストから画像へのモデル（例えばStableDiffusion[^22]）や、DreamBooth[^24]やLoRA[^13]のような対応するパーソナライゼーション技術の進歩により、誰もが手頃なコストで自分の想像力を高品質の画像に現すことができる。その後、生成された静止画像とモーションダイナミクスをさらに組み合わせる画像アニメーション技術が大いに求められている。本論文では、既存のパーソナライズされたテキスト画像モデルのほとんどを一度でアニメーション化する実用的なフレームワークを提案し、モデル固有のチューニングの手間を省く。提案するフレームワークの中核は、凍結されたテキストから画像へのモデルに、新たに初期化されたモーションモデリングモジュールを挿入し、妥当なモーションプリオアを抽出するためにビデオクリップでそれを訓練することである。一度訓練されれば、このモーション・モデリング・モジュールを注入するだけで、同じベースT2Iから派生したすべてのパーソナライズされたバージョンは、多様でパーソナライズされたアニメーション画像を生成するテキスト駆動モデルに容易になる。我々は、アニメの絵と現実的な写真にわたる、いくつかの一般的な代表的なパーソナライズされたテキストから画像へのモデルで評価を行い、我々の提案するフレームワークが、これらのモデルが、その出力のドメインと多様性を保持しながら、時間的に滑らかなアニメーションクリップを生成するのに役立つことを実証する。コードと事前に訓練された重みは、我々の[プロジェクトページ](https://animatediff.github.io/)で公開される予定である。

## 1 はじめに

近年、text-to-image(T2I)生成モデル[^17] [^21] [^22] [^25]は、高いビジュアル品質とテキスト駆動の制御性、すなわち、アーティストやアマチュアといった研究者ではないユーザーがAI支援によるコンテンツ制作を行うための敷居の低いエントリーポイントを提供することから、研究コミュニティ内外でかつてない注目を集めている。既存のT2I生成モデルの創造性をさらに刺激するために、DreamBooth[^24]やLoRA[^13]などの軽量なパーソナライゼーション手法がいくつか提案されており、RTX3080を搭載したノートPCなどのコンシューマグレードのデバイスを使用して、小規模なデータセット上でこれらのモデルをカスタマイズして微調整することができる。このようにして、ユーザーは、非常に低コストで、事前に訓練されたT2Iモデルに新しいコンセプトやスタイルを導入することができ、その結果、CivitAI[^4]やHuggingface[^8]のようなモデル共有プラットフォームで、アーティストやアマチュアによって投稿された多数のパーソナライズされたモデルが生まれる。

DreamBoothやLoRAで学習されたパーソナライズド・テキスト画像モデルは、その卓越した視覚的品質によって注目を集めることに成功しているが、それらの出力は静止画像である。つまり、時間的な自由度がないのだ。アニメーションの幅広い応用を考慮すると、我々は、既存のパーソナライズされたT2Iモデルのほとんどを、元の視覚的品質を保持しながら、アニメーション画像を生成するモデルに変えることができるかどうかを知りたい。最近の一般的なテキストから動画への生成アプローチ[^7] [^12] [^33]は、時間的モデリングを元のT2Iモデルに組み込み、動画データセット上でモデルをチューニングすることを提案している。しかし、パーソナライズされたT2Iモデルには困難が伴う。なぜなら、ユーザーは通常、繊細なハイパーパラメータのチューニング、パーソナライズされたビデオの収集、集中的な計算リソースに余裕がないからである。

この研究では、一般的な手法であるAnimateDiffを紹介する。AnimateDiffは、どのようなパーソナライズされたT2Iモデルに対してもアニメーション画像を生成できるようにするもので、モデル固有のチューニング作業を必要とせず、長期にわたって魅力的なコンテンツの一貫性を実現する。ほとんどのパーソナライズされたT2Iモデルは、同じベースモデル（Stable Diffusion[^22]など）から派生しており、すべてのパーソナライズされたドメインに対応する動画を収集することは現実的でないため、ほとんどのパーソナライズされたT2Iモデルを一度にアニメーション化できるモーションモデリングモジュールを設計することにした。具体的には、ベースとなるT2Iモデルにモーションモデリングモジュールを導入し、大規模なビデオクリップ[^1]上で微調整を行い、妥当なモーションプリオアを学習する。注目すべきは、ベースモデルのパラメーターがそのまま残っていることだ。微調整の後、我々は、導き出されたパーソナライズされたT2Iが、よく学習されたモーションプリオからも恩恵を受け、滑らかで魅力的なアニメーションを生成できることを実証した。つまり、モーションモデリングモジュールは、追加のデータ収集やカスタマイズされたトレーニングを行うことなく、対応するすべてのパーソナライズされたT2Iモデルをアニメーション化することができる。

DreamBooth[^24]とLoRA[^13]の代表的なアニメ絵と現実的な写真のモデルを用いて、AnimateDiffを評価した。特別なチューニングを行わなくても、よく訓練されたモーション・モデリング・モジュールを挿入することで、ほとんどのパーソナライズされたT2Iモデルを直接アニメーション化することができる。実際には、モーション・モデリング・モジュールが適切なモーション・プリオールを学習するためには、時間次元に沿ったバニラ・アテンションで十分であることもわかった。また、3Dカートゥーンや2Dアニメのような領域にも一般化できることを示す。このため、我々のAnimateDiffは、パーソナライズされたアニメーションのためのシンプルかつ効果的なベースラインを提供することができる。

## 2 先行研究

### Text-to-Image 拡散モデル

近年、T2I拡散モデルは、大規模なテキスト‐画像ペアデータ[^26]と拡散モデル[^5] [^11]のパワーの恩恵を受け、研究コミュニティ内外で多くの人気を集めている。その中で、GLIDE[^17]は、拡散モデルにテキスト条件を導入し、分類器ガイダンスがより視覚的に好ましい結果をもたらすことを実証した。DALLE-2[^21]は、CLIP[^19]結合特徴空間を介してテキストと画像の位置合わせを改善する。Imagen[^25]は、フォトリアリスティックな画像生成を実現するために、テキスト・コーパスで事前学習された大規模な言語モデル[^20]と拡散モデルのカスケードを組み込んでいる。潜在拡散モデル[^22]、すなわち安定拡散は、オートエンコーダの潜在空間でノイズ除去処理を実行することを提案し、生成画像の品質と柔軟性を維持しながら、必要な計算リソースを効果的に削減する。生成過程でパラメータを共有する上記の作品とは異なり、eDiff-I[^2]は、異なる合成段階に特化した拡散モデルのアンサンブルを学習した。我々の方法は、事前に訓練されたテキストから画像へのモデルに基づいて構築されており、あらゆるチューニングベースのパーソナライズされたバージョンに適応させることができる。

### パーソナライズ Text-to-Image モデル

多くの強力なT2I生成アルゴリズムがある一方で、大企業や研究機関しかアクセスできない大規模なデータと計算リソースが必要なため、個人ユーザーがモデルをトレーニングすることはまだ受け入れられない。そのため、事前に訓練されたT2Iモデルに、ユーザが新しいドメイン（主にユーザが収集した少数の画像によって表現される新しい概念やスタイル）を導入できるようにするための手法がいくつか提案されている[^6] [^9] [^10] [^14] [^16] [^24] [^27]。Textual Inversion[^9]は、各概念の単語埋め込みを最適化し、学習中に元のネットワークを凍結することを提案した。DreamBooth[^24]は、保存損失を規制としてネットワーク全体を微調整する別のアプローチである。カスタム拡散[^16]は、パラメータの小さなサブセットのみを更新し、閉形式の最適化によって概念の併合を可能にすることで、微調整の効率を向上させる。同時に、DreamArtist[^6]は、入力を単一の画像に縮小する。最近では、言語モデル適応のために設計された技術であるLoRA[^13]が、テキストから画像へのモデルの微調整に利用され、良好な視覚的品質を達成している。これらの方法は主にパラメータチューニングに基づいているが、いくつかの研究では、概念パーソナライゼーションのためのより一般的なエンコーダの学習も試みられている[^10] [^14] [^27]。

研究コミュニティでは、このようなパーソナライゼーションアプローチがある中、我々の研究はチューニングベースの手法、すなわちDreamBooth[^24]とLoRA[^13]にのみ焦点を当てている。

### パーソナライズ T2I アニメーション

この論文の設定は新しく提案されたものであるため、現在、これを対象とした研究はほとんどない。映像生成のために既存のT2Iモデルを時間構造で拡張することは一般的であるが、既存の研究[^7] [^12] [^15] [^28] [^31] [^33]は、元のT2Iモデルのドメイン知識を傷つけながら、ネットワーク内のパラメータ全体を更新している。最近、パーソナライズされたT2Iモデルのアニメーションへの応用がいくつか報告されている。例えば、Tune-a-Video[^31]は、わずかなアーキテクチャの変更とサブネットワークのチューニングによって、ワンショットビデオ生成タスクを解決している。Text2Video-Zero[^15]は、事前に訓練されたT2Iモデルを、事前に定義されたアフィン行列を与えられた潜在的なラッピングによってアニメーション化する、訓練不要の手法を導入している。我々の手法に近い最近の研究は、Align-Your-Latents[^3]であり、これはT2Iモデルで別々の時間レイヤーを学習するT2V（text-to-video）モデルである。我々の方法は、簡略化されたネットワーク設計を採用し、多くのパーソナライズされたモデルに対する広範な評価を通じて、パーソナライズされたT2Iモデルのアニメーション化におけるこのアプローチの有効性を検証する。

## 3 手法

本節では、まず[3.1節](#31-前置き)で、一般的なテキスト画像変換モデルと、そのパーソナライズド・バリアントに関する予備知識を紹介する。次に、[3.2節](#32-パーソナライズされたアニメーション)では、パーソナライズドアニメーションの定式化と本手法の動機を述べる。最後に、[3.3節](#33-モーションモデリングモジュール)では、AnimateDiffのモーションモデリングモジュールの実用的な実装について説明する。このモジュールは、様々なパーソナライズされたモデルをアニメーション化し、魅力的な合成を生成する。

### 3.1 前置き

#### 一般的な T2I 生成器

本研究では、一般的なT2I生成器として、広く使用されているテキストから画像へのモデルであるStable Diffusion（SD）を選択した。SDは潜在拡散モデル（LDM）[^22]に基づいており、大規模な画像データセットで事前に訓練されたVQ-GAN[^14]またはVQ-VAE[^29]として実装されたオートエンコーダ、すなわち$\mathcal E(\cdot)$と$\mathcal D(\cdot)$の潜在空間でノイズ除去処理を実行する。この設計は、高い視覚的品質を保ちながら、計算コストを削減するという利点をもたらす。潜在拡散ネットワークの学習中、入力画像$x_0$は最初にフローズンエンコーダによって潜在空間にマッピングされ、$z_0=\mathcal E(x_0)$が得られ、次にあらかじめ定義されたマルコフ過程$$q\left(z_t|z_{t-1}\right)=\mathcal N\left(z_t;\sqrt{1-\beta_t}z_{t-1},\beta_tI\right)\tag{1}$$によって$t=1,\cdots,T$（$T$は順拡散過程のステップ数）の摂動が加えられる。一連のハイパーパラメータ$β_t$は、各ステップにおけるノイズの強さを決定する。上記の反復プロセスは、閉形式で次のように定式化できる。$$z_t=\sqrt{\bar{\alpha_t}}z_0+\sqrt{1-\bar{\alpha_t}}\epsilon,\epsilon～\mathcal N(0,I)\tag{2}$$ここで、$\bar{\alpha_t}=\prod^t_{i=1}\alpha_t,\alpha_t=1-\beta_t$である。Stable Diffusionは、DDPM[^5]で提案されているバニラ訓練目的を採用し、次の式で表される。$$\mathcal L=\mathbb E_{\mathcal E\left(x_0\right),y,\epsilon～\mathcal N(0,I),t}\left[\left|\left|\epsilon-\epsilon_\theta\left(z_t,t,\tau_\theta\left(y\right)\right)\right|\right|^2_2\right]\tag{3}$$ここで、$y$は対応するテキスト記述であり、$\tau_\theta(\cdot)$は文字列をベクトル列にマッピングするテキストエンコーダである。

SDでは、$\tau_\theta(\cdot)$は、4つのダウンサンプル/アップサンプルブロックと1つの中間ブロックを組み込んだ修正UNet[^23]で実装され、その結果、ネットワークの潜在空間内で4つの分解能レベルが得られる。各解像度レベルは、2次元畳み込み層と自己およびCross-Attentionメカニズムを統合している。テキストモデル$τ_θ(\cdot)$はCLIP[^19]ViT-L/14テキストエンコーダを使って実装されている。

#### パーソナライズド画像生成

一般的な画像生成が進歩するにつれ、パーソナライズされた画像生成に注目が集まっている。DreamBooth[^24]とLoRA[^13]は、代表的で広く使われている2つのパーソナライゼーションアプローチである。事前に訓練されたT2Iモデルに新しいドメイン（新しい概念、スタイルなど）を導入するには、その特定のドメインの画像で微調整するのが簡単なアプローチである。しかし、正則化なしでモデルを直接チューニングすると、特にデータセットが小さい場合、過学習や壊滅的な忘却につながることが多い。この問題を克服するために、DreamBooth[^24]では、ターゲット領域を表す指標として珍しい文字列を使用し、オリジナルのT2Iモデルによって生成された画像を追加することでデータセットを拡張している。これらの正則化画像はインジケータなしで生成されるため、微調整中にモデルが稀な文字列を期待されるドメインに関連付けることを学習することができる。

一方、LoRA[^13]は、モデル重みの残差を微調整すること、つまり$W$の代わりに$\Delta W$を訓練することで、異なるアプローチをとっている。微調整後の重みは、$W^′＝W＋\alpha\Delta W$として計算される。ここで、$\alpha$は、調整プロセスの影響を調整するハイパーパラメータであるため、生成された結果をユーザーがより自由に制御できる。さらにオーバーフィッティングを回避し、計算コストを削減するために、$\Delta W\in\mathbb R^{m\times n}$は2つの低ランク行列、すなわち$\Delta W = AB^T$に分解される。ここで、$A\in\mathbb R^{m\times r},B\in\mathbb R^{n\times r},r\ll m,n$である。実際には、トランスフォーマーブロックの射影行列のみが調整され、LoRAモデルの学習と保存のコストがさらに削減される。一度学習したモデルのパラメータをすべて保存するDreamBoothに比べ、LoRAモデルは学習やユーザー間の共有が非常に効率的である。

### 3.2 パーソナライズされたアニメーション

![図2](2023-09-25-10-39-58.png)
図2：**AnimateDiffのパイプライン**。ベースとなるT2Iモデル（例えば、Stable Diffusion[^22]）が与えられた場合、我々の手法では、まず動画データセットに対して動きモデリングモジュールを学習させ、動きプリオールを抽出させる。この段階では、モーションモジュールのパラメータのみが更新されるため、ベースとなるT2Iモデルの特徴空間は維持される。推論時に、一度訓練されたモーションモジュールは、ベースとなるT2Iモデルに基づいて調整されたパーソナライズされたモデルをアニメーションジェネレータに変換し、反復的なノイズ除去処理によって、多様でパーソナライズされたアニメーション画像を生成することができる。

パーソナライズされた画像モデルをアニメーション化するには、通常、対応するビデオコレクションを使ってさらにチューニングする必要があり、より困難な作業となる。本節では、パーソナライズされたアニメーションを対象とする。パーソナライズされたT2Iモデッド、例えば、ユーザによって訓練されたDreamBooth[^24]やLoRA[^13]のチェックポイント、またはCivitAI[^4]やHuggingface[^8]からダウンロードされたチェックポイントが与えられた場合、目標は、元のドメイン知識と品質を保持しながら、訓練コストをほとんど、または全くかけずにアニメーションジェネレータにトランスフォーマーすることである。例えば、T2Iモデルが特定の2Dアニメスタイル用にパーソナライズされているとする。その場合、対応するアニメーション・ジェネレーターは、前景／背景の分割、キャラクターの体の動きなど、適切なモーションを持つそのスタイルのアニメーション・クリップを生成できるものでなければならない。

これを達成するために、1つの素朴なアプローチは、大規模なビデオデータセットから、時間を意識した構造を追加し、合理的なモーションプリオを学習することによって、T2Iモデル[^7] [^12] [^33]を膨らませることである。しかし、パーソナライズド・ドメインの場合、十分なパーソナライズド・ビデオを集めるにはコストがかかる。同時に、限られたデータでは、ソース・ドメインの知識喪失につながる。そのため、我々は、汎化可能な運動モデリングモジュールを別途訓練し、推論時にパーソナライズされたT2Iにプラグインすることを選択した。そうすることで、パーソナライズされたモデルごとに特定のチューニングを行うことを避け、事前に訓練された重みを変更しないことで、モデルの知識を保持することができる。このようなアプローチのもう1つの重要な利点は、以下の実験で検証されているように、一度モジュールが訓練されれば、特定のチューニングを必要とすることなく、同じベースモデルに基づいてパーソナライズされたT2Iに挿入できることである。これは、パーソナライズ処理によって、ベースとなるT2Iモデルの特徴空間がほとんど変更されないためであり、ControlNet[^32]でも実証されている。

### 3.3 モーションモデリングモジュール

![図3](2023-09-25-10-45-15.png)
図3：**モーションモジュールの詳細**。モジュールの挿入（左）： 私たちのモーションモジュールは、事前に訓練された画像レイヤーの間に挿入される。データバッチが画像レイヤーとモーションモジュールを通過するとき、その時間軸と空間軸は別々にバッチ軸に整形される。モジュールの設計（右）： 私たちのモジュールは、初期化ゼロの出力プロジェクトレイヤーを持つバニラ時間トランスフォーマーである。

#### ネットワーク拡張

オリジナルのSDは画像データのバッチしか処理できないため、バッチ×チャンネル×フレーム×高さ×幅の5次元ビデオテンソルを入力とする我々のモーションモデリングモジュールと互換性を持たせるためには、モデル拡張が必要である。これを実現するために、ビデオ拡散モデル[^12]に似たソリューションを採用する。具体的には、フレーム軸をバッチ軸に再形成し、ネットワークが各フレームを独立して処理できるようにすることで、元の画像モデルの各2D畳み込み層とアテンション層を、空間のみの擬似3D層にトランスフォーマー化する。上記とは異なり、新たに挿入されたモーション・モジュールは、各バッチのフレームにまたがって動作し、アニメーション・クリップの動きの滑らかさとコンテンツの一貫性を実現する。詳細は図3に示す。

#### モジュール設計

モーション・モデリング・モジュールのネットワーク設計では、フレームをまたいだ効率的な情報交換を可能にすることを目指している。これを実現するために、モーションモジュールの設計にはバニラ時間トランスフォーマーを選んだ。モーションモジュールのために他のネットワーク設計も実験してみたが、バニラ時間トランスフォーマーがモーションプリオールをモデル化するのに十分であることがわかったことは注目に値する。より良いモーションモジュールの探求は、今後の研究に委ねたい。

バニラの時間トランスフォーマーは、時間軸に沿って動作する複数の自己アテンション・ブロックから構成されている（図3）。我々のモーション・モジュールを通過するとき、特徴マップ$z$の空間次元の高さと幅はまずバッチ次元に整形され、フレームの長さでバッチ×高さ×幅のシーケンスになる。その後、再形成された特徴マップは投影され、いくつかの自己注意ブロックを通過する。$$z=\rm{Attention}\left(Q,K,V\right)=\rm{Softmax}\left(\frac{QK^T}{\sqrt d}\right)\cdot V\tag 4$$ここで、$Q＝W^Qz、K＝W^Kz、V＝W^Vz$は、リシェイプされた特徴マップの3つの投影である。この操作により、時間軸をまたいで同じ位置にある特徴間の時間的依存関係を捉えることができる。モーション・モジュールの受容野を拡大するために、U字型拡散ネットワークのすべての解像度レベルに挿入する。さらに、正弦波位置エンコーディング[^30]をセルフ・アテンション・ブロックに追加し、ネットワークがアニメーション・クリップ内の現在のフレームの時間的位置を認識できるようにした。トレーニング中に有害な影響を与えることなく我々のモジュールを挿入するために、時間トランスフォーマーの出力投影層をゼロ初期化する。これはControlNet[^32]によって検証された効果的な方法である。

#### 訓練目標

我々のモーション・モデリング・モジュールの学習プロセスは、潜在拡散モデル[^22]に似ている。サンプリングされたビデオデータ$x^{1:N}_0$は、まず事前に訓練されたオートエンコーダによってフレームごとに潜在コード$z^{1:N}_0$にエンコードされる。次に、定義された前方拡散スケジュール$z^{1:N}_t=\sqrt{\bar{\alpha_t}}z^{1:N}_0+\sqrt{1-\bar{\alpha_t}}\epsilon$を用いて潜在符号をノイズ化する。我々のモーション・モジュールで拡張した拡散ネットワークは、ノイズの入った潜在コードと対応するテキストプロンプトを入力とし、L2損失項によって励まされながら、潜在コードに加えられるノイズの強さを予測する。モーション・モデリング・モジュールの最終的な訓練目的は以下のようになる。$$\mathcal L=\mathbb E_{\mathcal E\left(x^{1:N}_0\right),y,\epsilon～\mathcal N(0,I),t}\left[\left|\left|\epsilon-\epsilon_\theta\left(z^{1:N}_t,t,\tau_\theta(y)\right)\right|\right|^2_2\right]\tag 5$$

最適化の間、ベースとなるT2Iモデルの事前学習された重みは、その特徴空間を変更しないように凍結されることに留意されたい。

## 4 実験

### 4.1 実装の詳細

#### 訓練

モーション・モデリング・モジュールをトレーニングするためのベースモデルとして、Stable Diffusion v1を選んだ。テキストと動画のペアデータセットであるWebVid-10M[^1]を使ってモーションモジュールを学習させた。データセットのビデオクリップは、まず4のストライドでサンプリングされ、次に256×256の解像度にリサイズされ、センタークロップされる。我々の実験によれば、256で学習したモジュールは、より高い解像度に一般化できる。そのため、トレーニングの効率と視覚的な品質のバランスが保たれる256をトレーニング解像度に選んだ。トレーニング用のビデオクリップの最終的な長さは16フレームに設定された。実験中に、ベースとなるT2Iモデルが学習されたオリジナルのスケジュールとはわずかに異なる拡散スケジュールを使用することで、より良い視覚品質を達成し、低飽和度やちらつきなどの不自然さを避けることができることを発見した。元のスケジュールを少し修正することで、モデルが新しいタスク（アニメーション）や新しいデータ分布にうまく適応できるようになるという仮説を立てた。そこで、$β_{start}=0.00085$、$β_{end}=0.012$の線形ベータ・スケジュールを使用したが、これは元のSDの訓練に使用したものとは若干異なる。

#### 評価

![表1](2023-09-25-10-58-25.png)
表1：評価に使用したパーソナライズドモデル。評価には、CivitAI[^4]のアーティストから提供された、2Dアニメーションからリアルな写真まで幅広い領域をカバーする代表的なパーソナライズドモデルをいくつか選びました。

本手法の有効性と汎用性を検証するために、アーティストがパーソナライズされたモデルを共有できる公開プラットフォームであるCivitAI[^4]から、代表的なパーソナライズされたStable Diffusionモデル（表1）をいくつか収集した。これらの選ばれたモデルのドメインは、アニメや2D漫画画像から現実的な写真まで多岐にわたり、我々の手法の能力を評価するための包括的なベンチマークを提供している。私たちのモジュールが訓練されると、私たちはそれをターゲットのパーソナライズされたモデルに差し込み、デザインされたテキストプロンプトとともにアニメーションを生成する。一般的なテキストプロンプトを使用しないのは、パーソナライズドモデルが特定のテキスト分布でのみ期待されるコンテンツを生成するためである。つまり、プロンプトは特定の形式を持つか、「トリガーワード」を含む必要がある。そこで、次のセクションでは、モデルのホームページで提供されているプロンプトの例を使用して、モデルの最高のパフォーマンスを得る。

### 4.2 定性的な結果

![図4](2023-09-25-10-48-39.png)
図4：**定性的な結果**。ここでは、我々のフレームワークのモーション・モデリング・モジュールで注入されたモデルによって生成された16のアニメーション・クリップを実演する。各行の2つのサンプルは、同じパーソナライズドT2Iモデルに属している。スペースの都合上、各アニメーションクリップから4フレームのみ抜粋している。各プロンプトに含まれる無関係なタグ、たとえば「masterpieces」「high quality」などは、わかりやすくするために省略している。

図4では、さまざまなモデルにおけるいくつかの定性的な結果を示している。スペースの都合上、各アニメーションクリップは4フレームしか表示されない。読者の皆様には、より良いビジュアル・クオリティのホームページを参照されることを強くお勧めする。図は、我々の手法が、高度に様式化されたアニメ（1段目）からリアルな写真（4段目）まで、多様なドメインにおいて、ドメイン知識を損なうことなく、パーソナライズされたT2Iモデルのアニメーション化に成功していることを示している。ビデオデータセットから学習されたモーションプリオールのおかげで、モーションモデリングモジュールは、テキストプロンプトを理解し、海の波の動き（3列目）やパラスキャットの足の動き（7列目）のように、各ピクセルに適切なモーションを割り当てることができる。また、この手法では、主要な被写体を前景と背景から区別することができ、生き生きとした臨場感を生み出すことができる。例えば、最初のアニメーションでは、キャラクターと背景の花が別々に、異なるスピードで、異なるぼかしの強さで動いている。

我々の定性的な結果は、様々な領域でパーソナライズされたT2Iモデルをアニメーション化するための我々のモーションモジュールの汎用性を実証している。私たちのモーションモジュールをパーソナライズドモデルに挿入することで、AnimateDiffは、多様で視覚的に魅力的でありながら、パーソナライズド領域に忠実な高品質のアニメーションを生成することができます。

### 4.3 ベースラインとの比較

![図5](2023-09-25-10-52-03.png)
図5：**ベースラインとの比較**。ベースライン（1行目、3行目）と我々の方法（2行目、4行目）の間のフレーム間のコンテンツの一貫性を定性的に比較する。ベースラインの結果が細かい粒度の一貫性に欠けるのに対し、我々の方法がより優れた時間的滑らかさを維持していることは注目に値する。

我々は、Text2Video-Zero[^15]と比較する。Text2Video-Zeroは、ネットワーク・インフレーションと潜在ワーピングを通して、ビデオ生成のためにT2Iモデルを拡張するための訓練不要のフレームワークである。Tune-a-VideoもパーソナライズされたT2Iアニメーションに利用できるが、追加の入力ビデオを必要とするため、比較の対象とはしていない。T2V-Zeroはパラメータチューニングに依存しないため、モデルの重みをパーソナライズされたものに置き換えることで、パーソナライズされたT2Iモデルをアニメーション化するために採用するのは簡単である。著者から提供されたデフォルトのハイパーパラメータを用いて、解像度512×512の16フレームのアニメーションクリップを生成する。

同じパーソナライズされたモデルで、同じプロンプト「A forbidden castle high up in the mountains, pixel art, intricate details2, hdr, intricate details」を使って、ベースラインと我々の手法のフレーム間のコンテンツの一貫性を定性的に比較する。私たちの手法とベースラインのきめ細かな詳細をより正確に示し、比較するために、図5の各フレームの左右下部に示されているように、各結果の同じ部分を切り取り、拡大した。

図に示すように、どちらの方法もパーソナライズド・モデルのドメイン知識を保持しており、フレームレベルの品質は同等である。しかし、T2V-ZEROの結果は、視覚的には似ているが、注意深く比較すると、フレーム間の細かい一貫性に欠ける。例えば、手前の岩（1列目）とテーブルの上のコップ（3列目）の形は時間とともに変化する。この矛盾は、アニメーションをビデオクリップとして再生すると、より顕著になる。対照的に、我々の方法は時間的に一貫したコンテンツを生成し、優れた滑らかさを維持する（2、4段目）。さらに、私たちのアプローチは、基礎となるカメラの動きによりよく沿った、より適切なコンテンツの変化を示し、私たちの手法の有効性をさらに際立たせている。この結果は妥当である。というのも、ベースラインは動きの事前分布を学習せず、ルールベースの潜在ワーピングによって視覚的一貫性を実現しているのに対し、我々の手法は大規模なビデオデータセットから知識を継承し、効率的な時間的アテンションによって時間的滑らかさを維持しているからである。

### 4.4 アブレーション研究

![表2](2023-09-25-11-00-26.png)
表2：アブレーション実験における3つの拡散スケジュール構成。Stable Diffusionを事前にトレーニングするスケジュールはSchedule Aである。

![図6](2023-09-25-11-02-01.png)
図6：**アブレーション実験**。安定拡散が事前に訓練されたスケジュールから、それぞれ異なる偏差レベルを持つ3つの拡散スケジュールを実験し、その結果を定性的に比較する。

我々は、トレーニング中の順方向拡散プロセスにおけるノイズスケジュールの選択を検証するために、アブレーション研究を実施した。前節では、少し変更した拡散スケジュールを使用することで、より良いビジュアルクオリティを達成できることを述べた。ここでは、先行研究で採用されている3つの代表的な拡散スケジュール（表2）を実験し、図6で対応する結果を視覚的に比較する。実験に使用した3つの拡散スケジュールのうち、スケジュールAはStable Diffusionの事前学習用スケジュール、スケジュールBは我々の選択であり、ベータ配列の計算方法がSDのスケジュールと異なる。スケジュールCはDDPM[^5]とDiT[^18]で使用されており、SDの事前学習用スケジュールとはより異なる。

図6に示すように、モーション・モデリング・モジュールのトレーニングにSDのオリジナル・スケジュール（スケジュールB）を使用した場合、アニメーション結果は、浅黒い色の不自然なものとなった。直感的には、事前学習と並行して拡散スケジュールを使用することは、モデルがすでに学習した特徴空間を保持するのに有効なはずなので、この現象は珍しい。スケジュールがトレーニング前のスケジュール（スケジュールAからスケジュールCへ）から外れるにつれて、生成されるアニメーションの彩度は上昇し、可動域は減少する。これら3つの構成の中で、私たちが選んだのは、映像のクオリティと動きの滑らかさの両方をバランスよく実現するものだ。

これらの観察に基づき、訓練段階で拡散スケジュールを少し修正することで、事前訓練されたモデルが新しいタスクやドメインに適応しやすくなるという仮説を立てた。我々のフレームワークの新しい学習目的は、拡散されたビデオシーケンスからノイズシーケンスを再構築することである。これは、ビデオシーケンスの時間構造を考慮することなく、フレーム単位で行うことができる。これは、T2Iモデルが事前に訓練された画像再構成タスクである。同じ拡散スケジュールを使用すると、モデルがまだ画像再構成に最適化されていると誤解する可能性があり、クロスフレームのモーションモデリングを担当するモーションモデリングモジュールの学習効率が低下し、その結果、アニメーションのちらつきや色のエイリアシングが発生しやすくなる。

## 5 限界と今後の課題

![図7](2023-09-25-11-05-15.png)
図7：**失敗例**。パーソナライズされた領域が現実的でない場合、我々の方法は適切なモーションを生成できない。

我々の実験では、パーソナライズされたT2Iモデルのドメインが、例えば2Dのディズニーアニメのような、現実的なものからかけ離れている場合に、ほとんどの失敗例が現れることが観察された（図7）。このような場合、アニメーション結果には明らかな不自然さがあり、適切な動きを作り出すことができない。これは、トレーニングビデオ（現実的なもの）とパーソナライズされたモデルの間に大きな分布のギャップがあるためだと考えられる。この問題に対する可能な解決策は、ターゲット・ドメインの複数のビデオを手動で収集し、モーション・モデリング・モジュールをわずかに微調整することである。

## 6 おわりに

本報告では、パーソナライズされたテキストから画像へのモデルアニメーションを可能にする実用的なフレームワークであるAnimateDiffを紹介する。このフレームワークは、既存のパーソナライズされたT2Iモデルのほとんどを、一旦アニメーションジェネレータに変えることを目的としている。我々は、T2Iベースで訓練された、シンプルに設計されたモーションモデリングモジュールを含む我々のフレームワークが、大規模なビデオデータセットから一般化可能なモーションプリオを抽出できることを実証する。一旦訓練されれば、我々のモーション・モジュールを他のパーソナライズされたモデルに挿入することで、対応するドメインに忠実でありながら、自然で適切なモーションを持つアニメーション画像を生成することができる。様々なパーソナライズされたT2Iモデルに対する広範な評価も、本手法の有効性と汎用性を検証している。このように、AnimateDiffは、パーソナライズされたアニメーションのためのシンプルかつ効果的なベースラインを提供し、幅広いアプリケーションに恩恵をもたらす可能性がある。

## 補遺

### A 追加の詳細

#### A.1 モデルの多様性

![図8](2023-09-25-11-07-26.png)
図8：**モデルの多様性**。ここでは、同じプロンプトとパーソナライズド・モデルで生成された結果の2つのグループを示し、AnimateDiffで拡張した後でも、パーソナライズド・ジェネレーターがその多様性を維持していることを示す。

図8では、同じモデルで同じプロンプトを使用した結果を示しており、我々の方法がオリジナルモデルの多様性を損なわないことを示している。

#### A.2 定性的な結果

![図9](2023-09-25-11-09-30.png)
![図10](2023-09-25-11-11-31.png)
図9, 10：**追加の定性的な結果**。私たちのフレームワークのモーション・モデリング・モジュールで注入されたモデルによって生成されたいくつかのアニメーション・クリップを示す。各プロンプトに含まれる無関係なタグ、例えば「masterpieces」、「high quality」などは、わかりやすくするために省略されている。

図9と図10では、異なるパーソナライズド・モデルに対する本手法の結果を示している。

## 参照 <!-- omit in toc -->

[^1]:Max Bain, Arsha Nagrani, Gul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738,2021.

[^2]:Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. ediffi: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

[^3]:Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis.Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023.

[^4]:Civitai. Civitai. [https://civitai.com/](https://civitai.com/), 2022.

[^5]:Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems, 34:8780–8794, 2021.

[^6]:Ziyi Dong, Pengxu Wei, and Liang Lin. Dreamartist: Towards controllable one-shot text-to-image generation via contrastive prompt-tuning. arXiv preprint arXiv:2211.11337, 2022.

[^7]:Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. arXiv preprint arXiv:2302.03011, 2023.

[^8]:Hugging Face. Hugging face. [https://huggingface.co/](https://huggingface.co/), 2022.

[^9]:Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel CohenOr. An image is worth one word: Personalizing text-toimage generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

[^10]:Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Designing an encoder for fast personalization of text-to-image models. arXiv preprint arXiv:2302.12228, 2023.

[^11]:Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[^12]:Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. arXiv preprint arXiv:2204.03458, 2022.

[^13]:Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[^14]:Xuhui Jia, Yang Zhao, Kelvin CK Chan, Yandong Li, Han Zhang, Boqing Gong, Tingbo Hou, Huisheng Wang, and Yu-Chuan Su. Taming encoder for zero fine-tuning image customization with text-to-image diffusion models. arXiv preprint arXiv:2304.02642, 2023.

[^15]:Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Text-toimage diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439, 2023.

[^16]:Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.

[^17]:Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[^18]:William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

[^19]:Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[^20]:Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.

[^21]:Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

[^22]:Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[^23]:Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net:Convolutional networks for biomedical image segmentation, 2015.

[^24]:Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch,
Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine
tuning text-to-image diffusion models for subject-driven
generation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 22500–
22510, 2023.

[^25]:Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.

[^26]:Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402, 2022.

[^27]:Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image generation without testtime finetuning. arXiv preprint arXiv:2304.03411, 2023.

[^28]:Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.

[^29]:Aaron van den Oord, Oriol Vinyals, and koray kavukcuoglu. Neural discrete representation learning. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

[^30]:Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[^31]:Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Weixian Lei, Yuchao Gu, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. arXiv preprint arXiv:2212.11565, 2022.

[^32]:Lvmin Zhang and Maneesh Agrawala. Adding conditionalcontrol to text-to-image diffusion models. arXiv preprint arXiv:2302.05543, 2023.

[^33]:Daquan Zhou, Weimin Wang, Hanshu Yan, Weiwei Lv, Yizhe Zhu, and Jiashi Feng. Magicvideo: Efficient video generation with latent diffusion models. arXiv preprint arXiv:2211.11018, 2022.
