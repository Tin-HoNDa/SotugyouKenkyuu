# ノイズ除去拡散確率モデル <!-- omit in toc -->

## 目次 <!-- omit in toc -->

- [概要](#概要)
- [1 はじめに](#1-はじめに)
- [2 背景](#2-背景)
- [3 拡散モデルとノイズ除去オートエンコーダー](#3-拡散モデルとノイズ除去オートエンコーダー)
  - [3.1 順工程と$L\_T$](#31-順工程とl_t)
  - [3.2 逆工程と$L\_{1:T-1}$](#32-逆工程とl_1t-1)
  - [3.3 データのスケーリング、逆工程デコーダーと$L\_0$](#33-データのスケーリング逆工程デコーダーとl_0)
  - [3.4 学習目的の簡素化](#34-学習目的の簡素化)
- [4 実験](#4-実験)
  - [4.1 サンプルの品質](#41-サンプルの品質)
  - [4.2 逆工程のパラメータ化と学習目的アブレーション](#42-逆工程のパラメータ化と学習目的アブレーション)
  - [4.3 階層符号化](#43-階層符号化)
    - [段階的非可逆圧縮](#段階的非可逆圧縮)
    - [プログレッシブ・ジェネレーション](#プログレッシブジェネレーション)
    - [自己回帰復号への接続](#自己回帰復号への接続)
  - [4.4 補間](#44-補間)
- [5 関連研究](#5-関連研究)
- [6 結論](#6-結論)
- [より大きな影響](#より大きな影響)
- [謝辞および資金提供の開示](#謝辞および資金提供の開示)
- [追加情報](#追加情報)
  - [LSUN](#lsun)
  - [段階的圧縮](#段階的圧縮)
- [A 拡張された導出](#a-拡張された導出)
- [B 実験内容](#b-実験内容)
- [C 関連研究についての議論](#c-関連研究についての議論)
- [D サンプル](#d-サンプル)
  - [追加サンプル](#追加サンプル)
  - [潜在構造と逆過程の確率性](#潜在構造と逆過程の確率性)
  - [粗から細への補間](#粗から細への補間)
- [画像集](#画像集)
- [参考](#参考)

## 概要

我々は、非平衡熱力学の考察から着想を得た潜在変数モデルの一種である拡散確率モデルを用いた高品質な画像合成結果を発表する。我々の最良の結果は、拡散確率モデルとランジュヴィンダイナミクスを用いたノイズ除去スコアマッチングとの間の新しい接続に従って設計された重み付き変分境界で学習することにより得られる。無条件の CIFAR10 データセットにおいて、我々は $9.46$ の Inception スコアと $3.17$ の最先端 FID スコアを得た。$256\times256$ の LSUN では、ProgressiveGAN と同様のサンプル品質が得られた。我々の実装は[こちら](https://github.com/hojonathanho/diffusion。)

## 1 はじめに

最近、あらゆる種類の深層生成モデルが、様々なデータモダリティにおいて高品質なサンプルを示している。生成的敵対ネットワーク（GAN）、自己回帰モデル、フロー、変分オートエンコーダ（VAE）は、印象的な画像や音声サンプルを合成してきた [^14] [^27] [^3] [^58] [^38] [^25] [^10] [^32] [^44] [^57] [^26] [^33] [^45]。また、GAN に匹敵する画像を生成する、エネルギーベースのモデリングやスコアマッチングの進歩も目覚ましい [^11] [^55]。

![図1](2023-07-10-10-19-15.png)
図1：**CelebA-HQ 256×256 で生成したサンプル（左）と無条件の CIFAR10（右）**

![図2](2023-07-10-10-20-25.png)
図2：**この研究で検討された有向グラフィカルモデル**

本稿では、拡散確率モデル [^53] の進展を紹介する。拡散確率モデル（ここでは簡潔に「拡散モデル」と呼ぶ）は、変分推論を用いて学習されたパラメータ化されたマルコフ連鎖であり、有限時間後にデータと一致するサンプルを生成する。この連鎖の遷移は拡散過程を逆行するように学習される。拡散過程はマルコフ連鎖であり、信号が破壊されるまで、サンプリングとは逆方向にデータにノイズを徐々に加える。拡散が少量のガウシアンノイズで構成される場合、サンプリングチェーンの遷移も条件付きガウシアンに設定すれば十分であり、特に簡単なニューラルネットワークのパラメータ化が可能になる。

拡散モデルは定義が簡単で訓練も効率的であるが、我々の知る限り、高品質のサンプルを生成できるという実証はなされていない。我々は、拡散モデルが実際に高品質なサンプルを生成できることを示し、時には他のタイプの生成モデルに関する発表結果よりも優れていることを示す（[4章](#4-実験)）。さらに、拡散モデルのあるパラメータ化によって、学習中の複数のノイズレベルにわたるノイズ除去スコアマッチングと、サンプリング中のアニールされたランジュバン動力学との等価性が明らかになることを示す（[3.2節](#32-逆工程とl_1t-1)）[^55] [^61]。我々はこのパラメタリゼーションを用いて最高のサンプル品質結果を得たので（[4.2節](#42-逆工程のパラメータ化と学習目的アブレーション)）、この等価性を我々の主要な貢献の一つと考える。

サンプルの品質にもかかわらず、我々のモデルは他の尤度ベースのモデルと比較して競争力のある対数尤度を持たない（しかし、我々のモデルは、エネルギーベースのモデルやスコアマッチング [^11] [^55] に対してアニールされた重要度サンプリングが生成することが報告されている大きな推定値よりも優れた対数尤度を持つ）。我々は、我々のモデルのロスレスコード長の大部分が、知覚できない画像の細部を記述するために消費されていることを発見した（[4.3節](#43-階層符号化)）。我々はこの現象について、非可逆圧縮の言語でより洗練された分析を行い、拡散モデルのサンプリング手順が、自己回帰モデルで通常可能なことを大幅に一般化したビット順序に沿った自己回帰復号に類似したプログレッシブ復号の一種であることを示す。

## 2 背景

拡散モデル [^53] は、$p_\theta(x_0)\coloneqq\int p_\theta(x_{0:T})dx_{1:T}$ という形の潜在変数モデルで、$x_1,\cdots,x_T$ はデータ $x_0～p(x_0)$ と同じ次元の潜在である。共同分布 $p_\theta(x_{0:T})$ は逆プロセスと呼ばれ、$p(x_T)=\mathcal{N}(x_T;0,I)$ から始まる学習されたガウス遷移を持つマルコフ連鎖として定義される。

$$\begin{align*}
  p_\theta(x_{0:T})&\coloneqq p(x_T)\prod^T_{t=1}p_\theta(x_{t-1}|x_t),\\
  p_\theta(x_{t-1}|x_t)&\coloneqq\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
\end{align*}\tag{1}$$

拡散モデルが他のタイプの潜在変数モデルと異なる点は、順過程または拡散過程と呼ばれる近似事後 $q(x_{1:T},|x_0)$ が、分散スケジュール $\beta_1,\cdots,\beta_T$ に従ってデータにガウスノイズを徐々に加えるマルコフ連鎖に固定されていることである。

$$\begin{align*}
  q(x_{1:T}|x_0)\coloneqq&\prod^T_{t=1}q(x_t|x_{t-1})\\
  q(x_t|x_{t-1})\coloneqq&\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}.\beta_t\bold{I})\\
\end{align*}\tag{2}$$

学習は、負の対数尤度に関する通常の変分境界を最適化することによって行われる。

$$\begin{align*}
  \mathbb{E}\left[-\log p_\theta(x_0)\right]&\leq\mathbb{E}_q\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]\\
  &=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\\
  &\eqqcolon L
\end{align*}\tag{3}$$

順プロセスの分散 $\beta_t$ は再パラメータ化 [^33] によって学習することも、ハイパーパラメータとして一定に保つこともでき、逆プロセスの表現力は、どちらの過程も $\beta_t$ が小さいときには同じ関数形をとる [^53] ため、$p_\theta (x_{t-1}|x_t)$ のガウス条件式の選択によって一部保証される。順プロセスの注目すべき特性は、任意のタイムステップ $t$ で $x_t$ を閉じた形でサンプリングできることである。 $\alpha_t\coloneqq1-\beta_t$ 、$\bar\alpha_t\coloneqq\prod^t_{s=1}\alpha_s$ とすると、以下の式が成り立つ。

$$q(x_t|x_0)=\mathcal{N}\left(x_t;\sqrt{\bar\alpha_t}x_0,\left(1-\bar\alpha_t\right)\bold{I}\right)\tag{4}$$

したがって、確率的勾配降下法を用いて $L$ のランダム項を最適化することにより、効率的な学習が可能となる。さらに、式(3)の $L$ を式(5)のように書き換えることで分散を減らすことができる（詳細は[付録A](#a-拡張された導出)を参照、用語のラベルは[第3節](#3-拡散モデルとノイズ除去オートエンコーダー)で使用）。

$$\mathbb{E}_q\left[D_\text{KL}\left(q\left(x_T|x_0\right)||p\left(x_T\right)\right)+\sum_{t>1}D_\text{KL}\left(q\left(x_{t-1}|x_t,x_0\right)||p_\theta\left(x_{t-1}|x_t\right)\right)-\log p_\theta\left(x_0,x_1\right)\right]\tag{5}$$

式(5)は、KLダイバージェンスを使って、$p_\theta(x_{t-1}|x_t)$ を前進過程の後置と直接比較するもので、$x_0$ を条件とすると扱いやすい。

$$q\left(x_{t-1}|x_t,x_0\right)=\mathcal{N}\left(x_{t-1};\tilde\mu_t\left(x_t,x_0\right),\tilde\beta_t\bold{I}\right)\tag{6}$$

ただし、

$$\begin{align*}\tilde\mu_t(x_t,x_0)&:=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t\\\tilde\beta_t&:=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t\end{align*}\tag{7}$$

その結果、式(5)のKL発散はすべてガウシアン間の比較であるため、高バリアンスのモンテカルロ推定ではなく、閉形式のRao-Blackwell化された方法で計算することができる。

## 3 拡散モデルとノイズ除去オートエンコーダー

拡散モデルは潜在変数モデルの制限されたクラスのように見えるかもしれないが、実装の自由度が大きい。順プロセスの分散 $\beta_t$ と、逆プロセスのモデルアーキテクチャとガウス分布のパラメータ化を選択しなければならない。我々の選択を導くために、拡散モデルとノイズ除去スコアマッチングとの間の新しい明示的な接続を確立し([3.2節](#32-逆工程とl_1t-1))、拡散モデルに対する単純化された重み付き変分境界目的([3.4節](#34-学習目的の簡素化))を導く。最終的に、我々のモデルデザインは単純さと実証結果によって正当化される（[4章](#4-実験)）。我々の議論は式(5)によって分類される。

### 3.1 順工程と$L_T$

我々は前方過程の分散 $\beta_t$ が再パラメータ化によって学習可能であることを無視し、代わりに定数に固定する（[4章](#4-実験)）。従って、我々の実装では、近似事後 $q$ は学習可能なパラメータを持たないので、$L_T$は学習中は定数であり、無視することができる。

### 3.2 逆工程と$L_{1:T-1}$

ここで、$1＜t≦T$ に対する $p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$ の選択について説明する。まず、$\Sigma_\theta(x_t,t)=\sigma^2_t\bold{I}$ を訓練していない時間依存定数に設定する。実験的には、$\sigma^2_t=\beta_t$ と $\sigma^2_t=\bar\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ はどちらも同じような結果であった。最初の選択は $x_0～\mathcal{N}(\bold{0,I})$ に対して最適であり、2番目の選択は $x_0$ を決定論的に1点に設定した場合に最適である。これらは、座標的に単位分散を持つデータに対する逆プロセス・エントロピーの上界と下界に対応する2つの極端な選択である [^53]。

第二に、平均 $\mu_\theta(x_t,t)$ を表現するために、我々は以下の $L_t$ の分析に動機づけられた特定のパラメータ化を提案する。$p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t),\sigma^2_t\bold{I})$ とすると、以下の式が成り立つ。ここで $C$ は $\theta$ に依存しない定数である。

$$L_{t-1}=\mathbb{E}_q\left[\frac{1}{2\sigma^2_t}\left|\left|\tilde\mu(x_t,x_0)-\mu_\theta(x_t,t)\right|\right|^2\right]+C\tag{8}$$

つまり、$\mu_\theta$ の最も簡単なパラメタリゼーションは、前進過程の事後平均である $\tilde\mu_t$ を予測するモデルであることがわかる。しかし、$\epsilon～\mathcal{N}(\bold{0,I})$ に対する $x_t(x_0,\epsilon)=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ として式(4)を再パラメータ化し、順過程の事後式(7)を適用することで、式(8)をさらに拡張することができる。

$$\begin{align*}L_{t-1}-C&=\mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma^2_t}\left|\left|\bar\mu_t\left(x_t(x_0,\epsilon),\frac{1}{\sqrt{\bar\alpha_t}}(x_t(x_0,\epsilon)-\sqrt{1-\bar\alpha_t}\epsilon)\right)-\mu_\theta(x_t(x_0,\epsilon),t)\right|\right|^2\right]\tag{9}\\&=\mathbb{E}_{x_0,\epsilon}\left[\frac{1}{2\sigma^2_t}\left|\left|\frac{1}{\sqrt{\alpha_t}}\left(x_t(x_0,\epsilon)-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\right)-\mu_\theta(x_t(x_0,\epsilon),t)\right|\right|^2\right]\tag{10}\end{align*}$$

![アルゴリズム1,2](2023-07-04-11-25-54.png)

式(10)から、$\mu_\theta$ は $x_t$ が与えられたときに $\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\alpha_t}}\epsilon\right)$ を予測しなければならないことがわかる。$x_t$ はモデルへの入力として利用できるので、$\epsilon_\theta$ は $x_t$ から $\epsilon$ を予測するための関数近似であるパラメータ化

$$\mu_\theta(x_t,t)=\tilde\mu_t\left(x_t,\frac{1}{\sqrt{\bar\alpha_t}}\left(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t)\right)\right)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)\tag{11}$$

を選択することができる。$x_{t-1}∼p_\theta(x_{t-1}|x_t)$ を標本化するには、$z∼\mathcal{N}(\bold{0,I})$ に対して、$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_tz$ を計算する。完全なサンプリング手順であるアルゴリズム2は、θをデータ密度の学習勾配とするランジュヴィン・ダイナミクスに似ている。さらに、パラメータ化(11)により、式(10)は

$$\mathbb{E}_{x_0,\epsilon}\left[\frac{\beta^2_t}{2\sigma^2_t\alpha_t(1-\bar\alpha_t)}\left|\left|\epsilon-\epsilon_\theta\left(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t\right)\right|\right|^2\right]\tag{12}$$

に単純化され、これは $t$ [^55] でインデックス化された複数のノイズスケールにわたるノイズ除去スコアマッチングに似ている。式(12)はランジュバン的逆過程(11)の変分境界(の1項)に等しいので、ノイズ除去スコアマッチングに似た目的を最適化することは、ランジュバン力学に似たサンプリング鎖の有限時間マージンを適合させるために変分推論を使うことと等価であることがわかる。

要約すると、逆プロセス平均関数近似器 $\mu_\theta$ を訓練して $\tilde\mu_t$ を予測することもできるし、そのパラメタリゼーションを修正することによって、それを訓練して $\epsilon$ を予測するように訓練することもできる($x_0$ を予測する可能性もあるが、これは実験の初期段階でサンプルの質を悪化させることがわかった)。我々は $\epsilon$ 予測パラメタリゼーションがランジュヴィン・ダイナミクスに似ており、拡散モデルの変分束縛を単純化し、ノイズ除去のスコアマッチングに似た目的になることを示した。とはいえ、これは $p_\theta(x_{t-1}|x_t)$ の別のパラメタリゼーションに過ぎないので、[4章](#4-実験)で、$\epsilon$ を予測することと$\tilde\mu_t$を予測することを比較するアブレーションで、その有効性を検証する。

### 3.3 データのスケーリング、逆工程デコーダーと$L_0$

画像データは $\{0,1,\cdots,255\}$ の整数からなり、$[-1, 1]$ に線形にスケーリングされると仮定する。これにより、ニューラルネットワークの逆プロセスは、標準的な正規事前分布 $p(x_T)$ から始まる一貫してスケーリングされた入力で動作することが保証される。離散対数尤度を得るために、逆プロセスの最後の項をガウシアン $\mathcal{N}(x_0;\mu_\theta(x_1,1),\sigma^2_1\bold{I})$

$$\begin{align*}\\
p_\theta(x_0|x_1)&=\prod^D_{i=1}\int^{\delta_+(x^i_0)}_{\delta_-(x^i_0)}\mathcal{N}(x;\mu^i_\theta(x_1,1),\sigma^2_1)dx\\
\delta_+(x)&=\left\{\begin{array}{l}\infin\space\space\space\space\space\space\space\space(x=1)\\x+\frac{1}{255}(x<1)\end{array}\right.\\
\delta_-(x)&=\left\{\begin{array}{l}-\infin\space\space\space\space\space(x=-1)\\x-\frac{1}{255}(x>-1)\end{array}\right.\end{align*}\tag{13}$$

から得られる独立離散デコーダーに設定する。ここで、$D$ はデータの次元数であり、$i$ の上付き添字は1つの座標の抽出を示す(その代わりに条件付き自己回帰モデルのような、より強力なデコーダーを組み込むのは簡単だが、それは今後の研究に譲る)。VAEデコーダや自己回帰モデル [^34] [^52] で使用される離散化された連続分布と同様に、ここでの我々の選択は、データにノイズを加えたり、スケーリング演算のヤコビアンを対数尤度に組み込んだりする必要がなく、変分境界が離散データのロスレスコード長であることを保証する。サンプリングの最後に、$\mu_\theta(x_1,1)$ をノイズレスで表示する。

### 3.4 学習目的の簡素化

上で定義した逆プロセスとデコーダーにより、式(12)と式(13)から導かれる項で構成される変分境界は、$\theta$ に関して明確に微分可能であり、学習に使用できる。しかし、我々は、$t$ が $1$ から $T$ の間で一様である変分境界

$$L_{simple}(\theta):=\mathbb{E}_{t,x_0,\epsilon}\left[\left|\left|\epsilon-\epsilon_\theta\left(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t\right)\right|\right|^2\right]\tag{14}$$

の以下の変種で訓練することが、サンプルの品質にとって有益であることを発見した（そして、実装がより簡単である）。$t=1$ の場合は、離散デコーダの定義(13)の積分を、$σ^2_1$ とエッジ効果を無視して、ビン幅を倍したガウス確率密度関数で近似した $L_0$ に対応する。$t>1$ の場合は、式(12)の重み付けなしバージョンに対応し、NCSNノイズ除去スコア・マッチング・モデルで使用される損失重み付けに類似している [^55]（前進過程の分散 $\beta_t$ は固定されているため、$L_T$ は現れない）。アルゴリズム1は、この単純化された目的での完全な学習手順を示している。

我々の単純化された目的(14)は式(12)の重み付けを捨てているので、標準的な変分境界 [^18] [^22] と比較して再構成の異なる側面を強調する重み付き変分境界である。特に、[4章](#4-実験)の拡散過程の設定により、単純化された目的は、小さな $t$ に対応する損失項の重み付けを下げる。これらの項は、非常に少量のノイズでデータをノイズ除去するようにネットワークを訓練するので、ネットワークが大きな $t$ 項でより困難なノイズ除去タスクに集中できるように、重み付けを下げることは有益である。この再重み付けがサンプルの品質向上につながることは、実験で確認できるだろう。

## 4 実験

サンプリング時に必要なニューラルネットワークの評価回数が先行研究 [^53] [^55] と一致するように、すべての実験で $T=1000$ に設定した。前進過程の分散は、$\beta_1=10^{-4}$ から $\beta_T=0.02$ まで直線的に増加する定数に設定した。これらの定数は、$[-1,1]$ にスケーリングされたデータに対して小さくなるように選択され、$x_T$ における信号対雑音比を可能な限り小さく保ちながら、逆方向過程と順方向過程がほぼ同じ関数形式を持つようにした（我々の実験では、1次元あたり $L_T=D_{KL}(q(x_T|x_0)||\mathcal{N}(\bold{0,I}))≈10^{-5}$ ビット）。

逆プロセスを表現するために、マスクされていない PixelCNN++ [^52] [^48] に類似した U-Net バックボーンを使用し、全体を通してグループ正規化を行う [^66]。パラメータは時間を超えて共有され、Transformer 正弦波位置埋め込み [^60] を使用してネットワークに指定される。我々は $16\times16$ の特徴マップ解像度で自己注意を使用する [^63] [^60]。詳細は[付録B](#b-実験内容)を参照。

### 4.1 サンプルの品質

![表1](2023-07-10-10-21-55.png)
表1：**CIFAR10 の結果（NLL の単位は bits/dim）**[^11] [^17] [^3] [^29] [^53] [^59] [^7] [^43] [^11] [^56] [^55] [^39] [^4] [^29]

表1は CIFAR10 における Inception スコア、FID スコア、負対数尤度（ロスレスコード長）である。FID スコアは 3.17 であり、我々の無条件モデルは、クラス条件付きモデルを含む文献のほとんどのモデルよりも優れたサンプル品質を達成している。我々の FID スコアは、標準的なプラクティスであるように、トレーニングセットに関して計算されています。テストセットに関して計算すると、スコアは 5.24 となり、これは文献にあるトレーニングセットの FID スコアの多くよりも優れている。

![図3](2023-07-10-10-25-21.png)
図3：**LSUN Church のサンプル（FID $=7.89$）**

![図4：](2023-07-10-10-26-27.png)
図4：**LSUN Bedroom のサンプル（FID $=4.90$）**

予想通り、真の変分境界でモデルを学習した方が、単純化された目的での学習よりも良いコード長が得られるが、後者の方が最も良いサンプル品質が得られることが分かる。CIFAR10 と CelebA-HQ の $256\times256$ サンプルについては図1を、LSUNの $256\times256$ サンプル [^71] については図3と図4を、詳細は[D章](#d-サンプル)を参照。

### 4.2 逆工程のパラメータ化と学習目的アブレーション

表2は、逆プロセスのパラメタリゼーションと学習目的（[3.2節](#32-逆工程とl_1t-1)）のサンプル品質効果を示す。我々は、$\tilde\mu$ を予測するベースラインオプションが、式(14)のような単純化された目的である、重み付けされていない平均二乗誤差の代わりに、真の変分境界で訓練された場合にのみうまく機能することを見出す。また、（パラメータ化された対角 $\Sigma_\theta(x_t)$ を変分境界に組み込むことによって）逆過程の分散を学習すると、固定分散と比較して不安定な学習となり、サンプルの質が低下することがわかる。我々が提案したように $\epsilon$ を予測することは、固定分散を用いた変分境界で学習した場合、$\tilde\mu$ を予測することとほぼ同等の性能を示すが、我々の単純化された目的で学習した場合は、はるかに良好である。

### 4.3 階層符号化

表1は、CIFAR10モデルのコード長も示している。これは、他の尤度ベースモデルで報告されているギャップと同程度であり、我々の拡散モデルが過学習していないことを示している（最近傍の可視化については[付録D](#d-サンプル)を参照）。それでも、我々のロスレスコード長は、エネルギーベースのモデルやアニールされた重要度サンプリングを用いたスコアマッチングで報告された有意な推定値[^11]よりも優れているが、他のタイプの尤度ベースの生成モデル[^7]には及ばない。

それにもかかわらず、我々のサンプルは高品質であるため、拡散モデルには誘導バイアスがあり、優れたロッシー・コンプレッサーになると結論づけられる。変分境界項 $L_1+\cdots+L_T$ をレート、$L_0$ をディストーションとして扱うと、最高品質のサンプルを使用した CIFAR10 モデルのレートは1.78ビット/dim、ディストーションは1.97ビット/dimとなり、これは0から255のスケールで平均二乗誤差0.95に相当する。可逆コード長の半分以上は、知覚できない歪みを表す。

#### 段階的非可逆圧縮

![アルゴリズム3,4](2023-07-10-10-28-11.png)

![図5](2023-07-10-10-29-02.png)
図5：**無条件CIFAR10テストセットのレート-歪み対時間**
歪みは $[0,255]$ スケールでの二乗平均平方根誤差で測定。詳細は表4を参照。

式(5)を反映したプログレッシブ・ロッシー・コードを導入することで、このモデルのレート・ディストーション動作をさらに詳しく調べることができる（プロシージャーへのアクセスを前提としたアルゴリズム3と4を参照のこと）。これは、最小ランダム符号化 [^19] [^20] のような、任意の分布 $p$ と $q$ に対して、平均しておよそ $D_{KL}(q(x)||p(x))$ ビットを使用してサンプル $x∼q(x)$ を送信できる手順へのアクセスを前提とする。$x_0 ∼ q(x_0)$ に適用すると、アルゴリズム3と4は、式(5)に等しい総期待符号長を用い て、$x_T,\cdots,x_0$ を、式(5)に等しい期待される全コード長で順に送信する。受信者は、任意の時刻 $t$ において、部分情報 $x_t$ が完全に利用可能であり、式(4)により漸進的に推定することができる(確率的再構成 $x_0 ∼ p_\theta (x_0|x_t)$ も有効だが、歪みの評価が難しくなるのでここでは考慮しない)。

$$x_0\approx\hat x_0=\left(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t)\right)/\sqrt{\bar\alpha_t}\tag{15}$$

図5は、CIFAR10 テストセットで得られたレート-歪みプロットを示している。各時間 $t$ において、歪みは平均二乗誤差 $\sqrt{\left|\left|x_0+\hat x_0\right|\right|^2/D}$ として計算され、レートは時間 $t$ においてこれまでに受信されたビットの累積数として計算される。歪みはレート-歪みプロットの低レート領域で急峻に減少し、ビットの大部分が実際に知覚できない歪みに割り当てられていることを示している。

#### プログレッシブ・ジェネレーション

![図6](2023-07-10-10-31-13.png)
図6：**無条件のCIFAR10プログレッシブ生成（左から右への経時的な $\hat x_0$）**
付録（図10および図14）に、経時的な拡張サンプルとサンプルの品質指標を掲載。

![図7](2023-07-10-10-33-46.png)
図7：同じ潜在を条件とする場合、CelebA-HQ $256×256$ のサンプルは高レベルの属性を共有する。右下の象限は $x_t$、他の象限は $p_\theta(x_0|x_t)$ からのサンプル。

また、ランダムビットからの漸進的解凍によって与えられる漸進的無条件生成過程を実行する。言い換えれば、アルゴリズム2を用いて逆プロセスからサンプリングしながら、逆プロセスの結果 $\hat x_0$ を予測する。図6と図10は、逆プロセスの過程における $\hat x_0$ のサンプル品質を示している。大規模な画像特徴が最初に現れ、細部が最後に現れる。図7は、様々な $t$ に対して $x_t$ を凍結した確率的予測 $x_0 ∼ p_\theta(x_0|x_t)$ を示したものである。$t$ が小さいときには、微細な特徴以外は保存され、$t$ が大きいときには、大規模な特徴のみが保存される。おそらくこれは概念圧縮 [^18] のヒントであろう。

#### 自己回帰復号への接続

なお、変分境界(5)は以下のように書き換えることができる（導出は[付録A](#a-拡張された導出)を参照）。

$$L=D_{KL}\left(q(x_T)||p(x_T)\right)+\mathbb{E}_q\left[\sum_{t\geq1}D_{KL}(q(x_{t-1}|x_t)||p_\theta(x_{t-1}|x_t))\right]+H(x_0)\tag{16}$$

ここで、拡散過程の長さTをデータの次元に設定し、$q(x_t|x_0)$ が最初の $t$ 個の座標をマスクアウトした $x_0$ 上に全ての確率質量を置くように（すなわち、$q(x_t|x_{t-1})$ が $t$ 番目の座標をマスクアウトするように）前進過程を定義し、$p(x_T)$ が空白の画像上に全ての質量を置くように設定し、議論のために、$p_\theta(x_{t-1}|x_t)$ を完全に表現可能な条件付き分布とすることを考える。これらの選択により、$D_{KL}(q(x_T)||p(x_T))=0$ となり、$D_{KL}(q(x_{t-1}|x_t)||p_\theta (x_{t-1}|x_t))$ を最小化することで、$p_\theta$ は座標 $t+1\cdots,T$ を変更せず、$t+1,\cdots,T$ が与えられたときに $t$ 番目の座標を予測するように訓練する。したがって、この特殊な拡散で $p_\theta$ をトレーニングすることは、自己回帰モデルをトレーニングすることになる。

したがって、ガウス拡散モデル(2)は、データ座標の並べ替えでは表現できない、一般化されたビット順序を持つ一種の自己回帰モデルと解釈できる。先行研究では、このような並べ替えは、サンプルの品質に影響を与える誘導バイアスを導入することが示されている [^38] ので、我々は、ガウスノイズは、マスキングノイズに比べて画像に追加する方がより自然であるかもしれないので、おそらくより大きな効果で、同様の目的を果たすガウス拡散を推測している。さらに、ガウス拡散の長さはデータの次元に等しいという制限を受けない。例えば、$T=1000$ を使用するが、これは我々の実験における $32×32×3$ または $256×256×3$ の画像の次元よりも小さい。ガウス拡散は、高速サンプリングのために短くすることも、モデルの表現力を高めるために長くすることもできる。

### 4.4 補間

![図8](2023-07-10-10-35-51.png)
図8：CelebA-HQ の $256\times256$ 画像を $500$ タイムステップの拡散で補間したもの。

$q$ を確率的エンコーダとして用いて潜像空間において元画像 $x_0,x'_0～q(x_0)$ を補間し、$x_t,x'_t～q(x_t|x_0)$ とし、線形補間された潜像 $\bar x_t=(1-\lambda)x_0+\lambda x'_0$ を逆プロセスによって画像空間に復号し、$\bar x_0～p(x_o|\bar x_t)$ とすることができる。事実上、図8（左）に描かれているように、我々は、ソース画像の線形補間バージョンからアーチファクトを除去するために逆のプロセスを使用している。$x_t$ と $x'_t$ が変わらないように、異なる $\lambda$ の値に対してノイズを固定した。図8（右）は CelebA-HQ の $256×256$ 画像（$t=500$）の補間と再構成である。この逆プロセスにより、高品質な再構成と、ポーズ、肌の色、髪型、表情、背景などの属性を滑らかに変化させる、もっともらしい補間結果が得られるが、眼鏡などは生成されない。$t$ を大きくすると、より粗く、より多様な補間となり、$t=1000$ で新しいサンプルが得られる（付録図9）。

## 5 関連研究

拡散モデルはフロー [^9] [^46] [^10] [^32] [^5] [^16] [^23] やVAE [^33] [^47] [^37] に似ているかもしれないが、拡散モデルは $q$ がパラメータを持たず、トップレベルの潜在量 $x_T$ がデータ $x_0$ との相互情報がほぼゼロになるように設計されている。我々の $\epsilon$ 予測逆過程パラメタリゼーションは、拡散モデルと、サンプリングのためのアニールされたランジュヴィン・ダイナミクスを用いた、複数のノイズレベルにわたるノイズ除去スコアマッチングとの間の接続を確立する [^55] [^56]。しかし、拡散モデルは素直な対数尤度評価が可能であり、学習手順は変分推論を用いてランジュバン動力学サンプラーを明示的に学習する（詳細は[付録C](#c-関連研究についての議論)）。この関連はまた、ある重み付けされた形のノイズ除去スコアマッチングが、ランジュバン的サンプラーを訓練するための変分推論と同じであるという逆の意味合いも持っている。マルコフ連鎖の遷移演算子を学習する他の方法としては、注入学習 [^2]、変分ウォークバック [^15]、生成確率ネットワーク [^1] などがある [^50] [^54] [^36] [^42] [^35] [^65]。

スコアマッチングとエネルギーベースモデリングとの間の既知の関連性によって、我々の研究は、エネルギーベースモデルに関する他の最近の研究 [^67] [^68] [^69] [^12] [^70] [^13] [^11] [^41] [^17] [^8] に影響を与える可能性がある。これは、アニーリングされた重要度サンプリング [^24] の1回の実行で、歪みペナルティに対してレートディストーション曲線が計算される方法を彷彿とさせる。我々の漸進的デコーディングの議論は、畳み込みDRAWや関連モデル [^18] [^40] に見ることができ、また、サブスケールの順序付けや自己回帰モデルのサンプリング戦略 [^38] [^64] などのより一般的なデザインにつながるかもしれない。

## 6 結論

拡散モデルを用いた高品質な画像サンプルを発表し、マルコフ連鎖、ノイズ除去スコアマッチングとアニールされたランジュバン動力学（およびその延長線上にあるエネルギーベースのモデル）、自己回帰モデル、プログレッシブ非可逆圧縮を訓練するための拡散モデルと変分推論との間に関連性を見出した。拡散モデルは画像データに対して優れた帰納的バイアスを持つようなので、他のデータモダリティや、他のタイプの生成モデルや機械学習システムの構成要素として、その有用性を調査することを楽しみにしている。

## より大きな影響

拡散モデルに関する我々の研究は、GAN、フロー、自己回帰モデルなどのサンプルの質を向上させる取り組みなど、他のタイプの深層生成モデルに関する既存の研究と同様の範囲を占めている。我々の論文は、拡散モデルをこの技法群における一般的に有用なツールにするための進歩を示すものであり、生成モデルがより広い世界に与えた（そしてこれから与えるであろう）影響を増幅させる役割を果たすかもしれない。

残念なことに、生成モデルの悪意ある利用法は数多く知られている。サンプル生成技術は、政治的な目的のために著名人の偽の画像やビデオを作成するために使用することができる。ソフトウェアツールが利用できるようになるずっと以前から、フェイク画像は手作業で作成されていたが、我々のような生成モデルはそのプロセスを簡単にする。幸いなことに、CNNで生成された画像には現在、検出を可能にする微妙な欠陥がある [^62] が、生成モデルの改良によって、これがより困難になるかもしれない。また、生成モデルは、学習対象となるデータセットのバイアスも反映する。多くの大規模なデータセットは、自動化されたシステムによってインターネットから収集されるため、特に画像がラベル付けされていない場合、これらのバイアスを除去することは困難である。このようなデータセットで訓練された生成モデルからのサンプルがインターネット上で拡散すれば、こうしたバイアスはさらに強化されることになる。

データが高解像度化し、世界的なインターネットトラフィックが増加するにつれて、多くの視聴者がインターネットにアクセスできるようにすることが重要になるかもしれない。我々の研究は、画像分類から強化学習まで、幅広い下流タスクのためのラベル付けされていない生データ上での表現学習に貢献するかもしれないし、拡散モデルは、アート、写真、音楽などの創造的な用途でも実行可能になるかもしれない。

## 謝辞および資金提供の開示

本研究は、ONR PECASE および NSF Graduate Research Fellowship（助成金番号DGE-1752814）の支援を受けた。GoogleのTensorFlow Research Cloud (TFRC) は Cloud TPU を提供した。

## 追加情報

### LSUN

LSUN データセットの FID スコアを表3に示す。印のスコアは StyleGAN2 がベースラインとして報告したもので、その他のスコアはそれぞれの著者が報告したものである。

![表3](2023-07-10-10-37-37.png)
表3：**LSUN $256×256$ データセットの FID スコア**[^27] [^28] [^30]

### 段階的圧縮

[4.3節](#43-階層符号化)での非可逆圧縮の議論は、概念実証に過ぎない。なぜなら、アルゴリズム3と4は、最小ランダム符号化 [^20] のような手順に依存しており、高次元データでは扱いにくいからである。これらのアルゴリズムは Sohl-Dickstein ら [^53] の変分境界(5)の圧縮解釈として役立つものであり、実用的な圧縮システムとしてはまだ不十分である。

![表4](2023-07-10-10-38-41.png)
表4：**無条件のCIFAR10テストセットのレート-ディストーション値**（図5に付随）

## A 拡張された導出

以下は、拡散モデルの分散変分境界である式(5)の導出である。この資料は Sohl-Dickstein ら [^53] のものである。ここでは完全性を期すためにのみ掲載する。

$$\begin{align*}
L&=\mathbb{E}_q\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]\tag{17}\\
&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\tag{18}\\
&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t>1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}-\log\frac{p_\theta(x_0|x_1)}{q(x_1|x_0)}\right]\tag{19}\\
&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t>1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}\cdot\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}-\log\frac{p_\theta(x_o|x_1)}{q(x_1|x_0)}\right]\tag{20}\\
&=\mathbb{E}_q\left[-\log\frac{p(x_T)}{q(x_T|x_0)}-\sum_{t>1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}-\log p_\theta(x_0|x_1)\right]\tag{21}\\
&=\mathbb{E}_q\left[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_\theta(x_{t-1}|x_t))-\log p_\theta(x_0|x_1)\right]\\\tag{22}\\
\end{align*}$$

以下は、$L$ の別バージョンである。$L$ の推定は扱いにくいが、[4.3節](#43-階層符号化)での議論には有用である。

$$\begin{align*}
L&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]\tag{23}\\
&=\mathbb{E}_q\left[-\log p(x_T)-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t)}\cdot\frac{q(x_{t-1})}{q(x_t)}\right]\tag{24}\\
&=\mathbb{E}_q\left[-\log\frac{p(x_T)}{q(x_T)}-\sum_{t\geq1}\log\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t)}-\log q(x_0)\right]\tag{25}\\
&=D_{KL}(q(x_T)||p(x_T))+\mathbb{E}_q\left[\sum_{t\geq1}D_{KL}(q(x_{t-1}|x_t)||p_\theta(x_{t-1}|x_t))\right]+H(x_0)\tag{26}\\
\end{align*}$$

## B 実験内容

このニューラルネット・アーキテクチャは、PixelCNN++ [^52] のバックボーンを踏襲しており、これは Wide ResNet [^72] をベースにした U-Net [^48] である。実装を簡単にするために、重みの正規化 [^49] をグループの正規化 [^66] に置き換えた。$32×32$ モデルは4つの特徴マップ解像度（$32×32～4×4$）を使用し、$256×256$ モデルは6つの解像度を使用する。すべてのモデルは、解像度レベルごとに2つの畳み込み残差ブロックと、畳み込みブロック間の $16×16$ の解像度の自己注意ブロックを持つ [^6]。拡散時間 $t$ は、各残差ブロックにTransformer正弦波位置埋め込み [^60] を加えることで指定される。CIFAR10 モデルのパラメータは 3,570万個、LSUN モデルと CelebA-HQ モデルのパラメータは 1億1,400万個である。また、フィルター数を増やすことで、約2億5,600万個のパラメーターを持つ LSUN ベッドルームモデルの大規模なバリエーションも訓練した。

すべての実験に TPU v3-8（8個の V100 GPU と同様）を使用した。我々の CIFAR モデルは、バッチサイズ128で毎秒21ステップで学習し（80万ステップで完了するまで学習するのに10.6時間）、256枚の画像のバッチをサンプリングするのに17秒かかる。我々の CelebA-HQ / LSUN (2562)モデルは、バッチサイズ64で毎秒2.2ステップで学習し、128枚の画像のバッチをサンプリングするのに300秒かかる。CelebA-HQ で50万ステップ、LSUN Bedroom で240万ステップ、LSUN Cat で180万ステップ、LSUN Church で120万ステップの学習を行った。より大きなLSUN Bedroom モデルは、115万ステップ学習させた。

ネットワークのサイズをメモリの制約内に収めるために初期段階でハイパーパラメータを選択した以外は、ハイパーパラメータ探索の大部分を CIFAR10 のサンプル品質に合わせて最適化し、その結果を他のデータセットに引き継いだ。

- $L_T\approx0$ となるように制約された定数、線形、2次式のスケジュールから $\beta_t$ のスケジュールを選んだ。sweepなしで $T=1000$ とし、$β_1=10^{-4}$ から $β_T=0.02$ までの線形スケジュールを選択した。
- $\{0.1,0.2,0.3,0.4\}$ の値をsweepして、CIFAR10 のドロップアウト率を $0.1$ に設定した。CIFAR10 上でドロップアウトを行わないと、正則化されていない PixelCNN++ [^52] におけるオーバーフィッティングのアーティファクトを彷彿とさせる、より貧弱なサンプルが得られる。他のデータセットのdropout率は、sweepせずにゼロに設定した。
- CIFAR10 のトレーニングでは、ランダムな水平フリップを使用した。フリップあり、フリップなしの両方のトレーニングを試したところ、フリップありの方がサンプルの質がわずかに向上することがわかった。また、LSUN Bedroom 以外のデータセットでは、ランダムな水平フリップを使用した。
- 実験の初期段階で Adam [^31] と RMSProp を試し、前者を選択した。ハイパーパラメータは標準値のままとした。学習率を $2×10^{-4}$ に設定してもsweepは行わず、$256×256$ の画像では $2×10^{-5}$ に下げたが、学習率を大きくすると学習が不安定になるようだ。
- バッチサイズは CIFAR10 では $128$、それ以上の画像では $64$ に設定した。これらの値を超えるsweepは行っていない。
- 減衰係数を $0.9999$ としたモデルパラメータの EMA を使用した。この値を超える掃引は行わなかった。

最終的な実験は1度学習され、サンプルの品質については学習を通して評価された。サンプルの品質スコアと対数尤度は、学習過程における最小 FID 値で報告される。CIFAR10 では、OpenAI [^51] と TTUR [^21] のリポジトリにあるオリジナルのコードを用いて、それぞれ $50000$ サンプルの Inception と FID のスコアを計算した。LSUN 上で、StyleGAN2 [^30] リポジトリのコードを使用して、$50000$ サンプルの FID スコアを計算した。 CIFAR10 と CelebA-HQ は [TensorFlow Datasets](https://www.tensorflow.org/datasets) から提供されたものをロードし、LSUN は StyleGAN のコードを用いて作成した。データセットの分割（またはその欠如）は、生成モデリングの文脈での使い方を紹介した論文では標準的なものである。すべての詳細はソースコードリリースに記載されている。

## C 関連研究についての議論

我々のモデル・アーキテクチャ、フォワード・プロセス、定義、事前分布は、サンプルの質を向上させる微妙だが重要な点で、NCSN [^55] [^56] とは異なっている。特に、学習後に潜在変数モデルを追加するのではなく、サンプラーを直接潜在変数モデルとして学習する。詳細は以下のとおりである。

1. self-attention 付き U-Net を使用する。 NCSN は、拡張畳み込みによる RefineNet を使用する。正規化層 (NCSNv1) や出力層 (v2) のみではなく、Transformer の正弦波位置埋め込みを追加することで、すべての層を $t$ に条件付ける。
2. 拡散モデルは、ノイズを加えても分散が大きくならないように、（$\sqrt{1-\beta_t}$ のファクターで）順プロセスのステップごとにデータをスケールダウンし、ニューラルネットの逆プロセスに一貫してスケールされた入力を提供する。 NCSNはこのスケーリングファクターを省略している。
3. NCSNとは異なり、我々の前進過程はシグナルを破壊し（$D_{KL}(q(x_T|x_0)||\mathcal{N}(\bold{0,I}))\approx0$）、$x_T$ の事前分布と集約事後分布の間の密接な一致を保証する。また、NCSN とは異なり、我々の $\beta_t$ は非常に小さいので、条件付きガウシアンによるマルコフ連鎖によって前進過程が可逆的であることが保証される。いずれも、サンプリング時に分布がずれるのを防ぐ。
4. 我々のランジュバン的サンプラーは、前進過程の $\beta_t$ から厳密に導き出された係数（学習率、ノイズスケールなど）を持つ。したがって、我々の学習手順は、$T$ ステップ後のデータ分布に一致するようにサンプラーを直接学習させる。つまり、変分推論を用いてサンプラーを潜在変数モデルとして学習するのである。対照的に、NCSNのサンプラー係数はポストホックに手作業で設定され、その学習手順はサンプラーの品質指標を直接最適化することを保証していない。

## D サンプル

### 追加サンプル

図11、図13、図16、図17、図18、図19は、CelebA-HQ、CIFAR10、LSUN データセットで学習した拡散モデルの未修正サンプルである。

### 潜在構造と逆過程の確率性

サンプリング中、事前分布 $x_T∼\mathcal{N}(\bold{0,I})$ とランジュヴィン・ダイナミクスはともに確率的である。2つ目のノイズ源の重要性を理解するために、CelebA $256×256$ データセットについて、同じ中間潜像を条件とする複数の画像をサンプリングした。図7は、$t\in{1000,750,500,250}$ の潜在 $x_t$ を共有する逆過程 $x_\theta∼p_\theta(x_0|x_t)$ からの複数のドローを示している。これを実現するために、先行からの最初のドローから逆連鎖を1回実行する。中間のタイムステップで、チェーンは複数の画像をサンプリングするために分割される。$x_{T=1000}$ で事前にドローした後にチェーンを分割すると、サンプルは大きく異なる。しかし、より多くのステップを経てチェーンが分割されると、サンプルは性別、髪の色、眼鏡、彩度、ポーズ、表情などの高レベルの属性を共有する。このことは、$\times750$ のような中間潜在は、知覚できないにもかかわらず、これらの属性を符号化していることを示している。

### 粗から細への補間

図9は、1組のCelebA 256×256画像間の補間を示している。拡散ステップの数を増やすと、元画像の構造がより破壊され、モデルが逆プロセスで補完する。これにより、細かい粒度でも粗い粒度でも補間することができる。拡散ステップが $0$ の限界の場合、補間はピクセル空間でソース画像を混合する。一方、拡散ステップが1000回を超えると、ソース情報は失われ、補間は新しいサンプルとなる。

![図9](2023-07-10-10-42-44.png)
図9：**潜在混合前の拡散ステップ数を変化させる粗いものから細かいものへの補間**

## 画像集

![図10](2023-07-10-10-44-16.png)
図10：**無条件のCIFAR10プログレッシブ・サンプリングの経時的品質**

![図11](2023-07-10-10-45-48.png)
図11：**CelebA-HQ $256×256$ 生成サンプル**

![図12](2023-07-10-10-47-09.png)
図12：**CelebA-HQ の $256×256$ 最近傍画像**
顔を囲む $100×100$ のクロップで計算される。一番左の列が生成されたサンプルで、残りの列が学習セットの最近傍である。

![図13](2023-07-10-10-48-34.png)
図13：**無条件のCIFAR10生成サンプル**

![図14](2023-07-10-10-49-16.png)
図14：**無条件のCIFAR10プログレッシブ生成**

![図15](2023-07-10-10-50-19.png)
図15：**無条件CIFAR10最近傍**
生成されたサンプルが一番左の列で、学習セットの最近傍が残りの列である。

![図16](2023-07-10-10-51-24.png)
図16：**LSUN Church が生成したサンプル（FID $=7.89$）**

![図17](2023-07-10-10-52-38.png)
図17：**LSUN Bedroom が生成したサンプル（FID $=4.90$）**

![図18](2023-07-10-10-53-49.png)
図18：**LSUN Bedroom の小さいモデルが生成したサンプル（FID $=6.36$）**

![図19](2023-07-10-10-54-53.png)
図19：**LSUN Cat が生成したサンプル（FID $=19.75$）**

## 参考

[^1]: Guillaume Alain, Yoshua Bengio, Li Yao, Jason Yosinski, Eric Thibodeau-Laufer, Saizheng Zhang, and Pascal Vincent. GSNs: generative stochastic networks. Information and Inference: A Journal of the IMA, 5(2):210–249, 2016.

[^2]: Florian Bordes, Sina Honari, and Pascal Vincent. Learning to generate samples from noise through infusion training. In International Conference on Learning Representations, 2017.

[^3]: Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.

[^4]: Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull, Yuan Cao, and Yoshua Bengio. Your GAN is secretly an energy-based model and you should use discriminator driven latent sampling. arXiv preprint arXiv:2003.06060, 2020.

[^5]: Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differentialequations. In Advances in Neural Information Processing Systems, pages 6571–6583, 2018.

[^6]: Xi Chen, Nikhil Mishra, Mostafa Rohaninejad, and Pieter Abbeel. PixelSNAIL: An improved autoregressive generative model. In International Conference on Machine Learning, pages 863–871, 2018.

[^7]: Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparsetransformers. arXiv preprint arXiv:1904.10509, 2019.

[^8]: Yuntian Deng, Anton Bakhtin, Myle Ott, Arthur Szlam, and Marc’Aurelio Ranzato. Residual energy-based models for text generation. arXiv preprint arXiv:2004.11714, 2020.

[^9]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

[^10]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. arXiv preprint arXiv:1605.08803, 2016.

[^11]: Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. In Advances in Neural Information Processing Systems, pages 3603–3613, 2019.

[^12]: Ruiqi Gao, Yang Lu, Junpei Zhou, Song-Chun Zhu, and Ying Nian Wu. Learning generative ConvNets via multi-grid modeling and sampling. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 9155–9164, 2018.

[^13]: Ruiqi Gao, Erik Nijkamp, Diederik P Kingma, Zhen Xu, Andrew M Dai, and Ying Nian Wu. Flow
contrastive estimation of energy-based models. In Proceedings of the IEEE/CVF Conference on ComputerVision and Pattern Recognition, pages 7518–7528, 2020.

[^14]: Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.

[^15]: Anirudh Goyal, Nan Rosemary Ke, Surya Ganguli, and Yoshua Bengio. Variational walkback: Learning a transition operator as a stochastic recurrent net. In Advances in Neural Information Processing Systems, pages 4392–4402, 2017.

[^16]: Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. FFJORD: Free-form continuous dynamics for scalable reversible generative models. In International Conference on Learning Representations, 2019.

[^17]: Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. In International Conference on Learning Representations, 2020.

[^18]: Karol Gregor, Frederic Besse, Danilo Jimenez Rezende, Ivo Danihelka, and Daan Wierstra. Towards conceptual compression. In Advances In Neural Information Processing Systems, pages 3549–3557, 2016.

[^19]: Prahladh Harsha, Rahul Jain, David McAllester, and Jaikumar Radhakrishnan. The communication complexity of correlation. In Twenty-Second Annual IEEE Conference on Computational Complexity (CCC’07), pages 10–23. IEEE, 2007.

[^20]: Marton Havasi, Robert Peharz, and José Miguel Hernández-Lobato. Minimal random code learning: Getting bits back from compressed model parameters. In International Conference on Learning Representations, 2019.

[^21]: Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In Advances in Neural Information Processing Systems, pages 6626–6637, 2017.

[^22]: Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-VAE: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations, 2017.

[^23]: Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel. Flow++: Improving flow-based generative models with variational dequantization and architecture design. In International Conference on Machine Learning, 2019.

[^24]: Sicong Huang, Alireza Makhzani, Yanshuai Cao, and Roger Grosse. Evaluating lossy compression rates of deep generative models. In International Conference on Machine Learning, 2020.

[^25]: Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video pixel networks. In International Conference on Machine Learning, pages 1771–1779, 2017.

[^26]: Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron van den Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. In International Conference on Machine Learning, pages 2410–2419, 2018.

[^27]: Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

[^28]: Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4401-4410, 2019.

[^29]: Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676v1, 2020.

[^30]: Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8110–8119, 2020.

[^31]: Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.

[^32]: Diederik P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. In Advances in Neural Information Processing Systems, pages 10215–10224, 2018.

[^33]: Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.

[^34]: Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In Advances in Neural Information Processing Systems, pages 4743–4751, 2016.

[^35]: John Lawson, George Tucker, Bo Dai, and Rajesh Ranganath. Energy-inspired models: Learning with sampler-induced distributions. In Advances in Neural Information Processing Systems, pages 8501–8513, 2019.

[^36]: Daniel Levy, Matt D. Hoffman, and Jascha Sohl-Dickstein. Generalizing Hamiltonian Monte Carlo with neural networks. In International Conference on Learning Representations, 2018.

[^37]: Lars Maaløe, Marco Fraccaro, Valentin Liévin, and Ole Winther. BIVA: A very deep hierarchy of latent variables for generative modeling. In Advances in Neural Information Processing Systems, pages 6548–6558, 2019.

[^38]: Jacob Menick and Nal Kalchbrenner. Generating high fidelity images with subscale pixel networks and multidimensional upscaling. In International Conference on Learning Representations, 2019.

[^39]: Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.

[^40]: Alex Nichol. VQ-DRAW: A sequential discrete VAE. arXiv preprint arXiv:2003.01599, 2020.

[^41]: Erik Nijkamp, Mitch Hill, Tian Han, Song-Chun Zhu, and Ying Nian Wu. On the anatomy of MCMC-based maximum likelihood learning of energy-based models. arXiv preprint arXiv:1903.12370, 2019.

[^42]: Erik Nijkamp, Mitch Hill, Song-Chun Zhu, and Ying Nian Wu. Learning non-convergent non-persistent short-run MCMC toward energy-based model. In Advances in Neural Information Processing Systems, pages 5233–5243, 2019.

[^43]: Georg Ostrovski, Will Dabney, and Remi Munos. Autoregressive quantile networks for generative modeling. In International Conference on Machine Learning, pages 3936–3945, 2018.

[^44]: Ryan Prenger, Rafael Valle, and Bryan Catanzaro. WaveGlow: A flow-based generative network for speech synthesis. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3617–3621. IEEE, 2019.

[^45]: Ali Razavi, Aaron van den Oord, and Oriol Vinyals. Generating diverse high-fidelity images with VQ-VAE-2. In Advances in Neural Information Processing Systems, pages 14837–14847, 2019.

[^46]: Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pages 1530–1538, 2015.

[^47]: Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In International Conference on Machine Learning, pages 1278–1286, 2014.

[^48]: Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.

[^49]: Tim Salimans and Durk P Kingma. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. In Advances in Neural Information Processing Systems, pages 901–909, 2016.

[^50]: Tim Salimans, Diederik Kingma, and Max Welling. Markov Chain Monte Carlo and variational inference: Bridging the gap. In International Conference on Machine Learning, pages 1218–1226, 2015.

[^51]: Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, pages 2234–2242, 2016.

[^52]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma. PixelCNN++: Improving the PixelCNN with discretized logistic mixture likelihood and other modifications. In International Conference on Learning Representations, 2017.
[^53]: Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265, 2015.

[^54]: Jiaming Song, Shengjia Zhao, and Stefano Ermon. A-NICE-MC: Adversarial training for MCMC. In Advances in Neural Information Processing Systems, pages 5140–5150, 2017.

[^55]: Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, pages 11895–11907, 2019.

[^56]: Yang Song and Stefano Ermon. Improved techniques for training score-based generative models. arXiv preprint arXiv:2006.09011, 2020.

[^57]: Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. WaveNet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.

[^58]: Aaron van den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks. International Conference on Machine Learning, 2016.

[^59]: Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with PixelCNN decoders. In Advances in Neural Information Processing Systems, pages 4790–4798, 2016.

[^60]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, pages 5998–6008, 2017.

[^61]: Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661–1674, 2011.

[^62]: Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot...for now. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020.

[^63]: Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7794–7803, 2018.

[^64]: Auke J Wiggers and Emiel Hoogeboom. Predictive sampling with forecasting autoregressive models. arXiv preprint arXiv:2002.09928, 2020.

[^65]: Hao Wu, Jonas Köhler, and Frank Noé. Stochastic normalizing flows. arXiv preprint arXiv:2002.06707, 2020.

[^66]: Yuxin Wu and Kaiming He. Group normalization. In Proceedings of the European Conference on Computer Vision (ECCV), pages 3–19, 2018.

[^67]: Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu. A theory of generative convnet. In International Conference on Machine Learning, pages 2635–2644, 2016.

[^68]: Jianwen Xie, Song-Chun Zhu, and Ying Nian Wu. Synthesizing dynamic patterns by spatial-temporal generative convnet. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7093–7101, 2017.

[^69]: Jianwen Xie, Zilong Zheng, Ruiqi Gao, Wenguan Wang, Song-Chun Zhu, and Ying Nian Wu. Learning descriptor networks for 3d shape synthesis and analysis. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8629–8638, 2018.

[^70]: Jianwen Xie, Song-Chun Zhu, and Ying Nian Wu. Learning energy-based spatial-temporal generative convnets for dynamic patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

[^71]: Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. LSUN: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015.

[^72]: Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.
