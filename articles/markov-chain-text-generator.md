---
title: "「しかのこのこのここしたんたん」をマルコフモデルで考えた時、生成される確率を実感できるデモを構築した"
slug: "markov-chain-text-generator"
published: true
type: "tech"
topics: ["Python", "Streamlit", "自然言語処理", "マルコフ連鎖"]
emoji: "🦌"
---


最近、「しかのこのこのここしたんたん」というリズミカルなフレーズが巷で話題です。このフレーズがマルコフモデルにより生成されると考えた時、非常に低い確率であることを体験できる[デモサイト](https://shikanoko.streamlit.app)を構築してみました。ぜひ、実際に体験してください。

## マルコフモデルとは

マルコフ連鎖（マルコフモデル）とは、状態の遷移が確率的に決定されるモデルのことです。言い換えれば、過去の状態にのみ基づいて未来の状態が決まるという概念です。この特性を利用して、自然言語処理の分野では文章やテキストの生成に活用されています。

![](/images/shikanoko_app.png)

## 「しかのこのこのここしたんたん」での具体例
### このフレーズを選んだ理由
このフレーズは短く、同じ文字や音が繰り返されているため、遷移確率を計算するのに適しています。初学者にも直感的に理解しやすい例と言えます。

### 状態遷移
例えば、「しかのこのこのここしたんたん」というテキストを入力として、マルコフモデルを構築すると次のような遷移確率が得られます。

**マルコフモデルに基づく遷移確率:**

* **遷移元: 'し'**
    * → 'か': 0.50
    * → 'た': 0.50
* **遷移元: 'か'**
    * → 'の': 1.00
* **遷移元: 'の'**
    * → 'こ': 1.00
* **遷移元: 'こ'**
    * → 'の': 0.50
    * → 'こ': 0.25
    * → 'し': 0.25
* **遷移元: 'た'**
    * → 'ん': 1.00
* **遷移元: 'ん'**
    * → 'た': 1.00

このモデルに基づいてテキストを生成すると、「したんたんたんたんたんたんた」や「しかのここしかのこのこのこし」のような、テキストが確率的に多く生成されます。
今回、「しかのこのこのここしたんたん」のフレーズは非常に低い確率であることが分かります。

## デモの作成方法

実際にこのモデルを用いてリアルタイムにテキストを生成し、さらに音声合成エンジンを使って音声に変換するインタラクティブなデモアプリケーションをStreamlitで作成します。デモは[こちら](https://shikanoko.streamlit.app)で公開していますので、ぜひ試してみてください。

## コード

以下がデモアプリケーションのソースコードです。

```python
import streamlit as st
from collections import defaultdict
import random
from gtts import gTTS
import io
import time
import base64
from gtts.tts import gTTSError

def create_markov_model(text):
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - 1):
        current_char = text[i]
        next_char = text[i + 1]
        model[current_char][next_char] += 1
    return model

def calculate_transition_probabilities(model):
    probabilities = {}
    for char, transitions in model.items():
        total = sum(transitions.values())
        probabilities[char] = {next_char: count / total for next_char, count in transitions.items()}
    return probabilities

def generate_text(probabilities, start_char, length=14):
    result = start_char
    current_char = start_char
    for _ in range(length - 1):
        if current_char in probabilities:
            next_char = random.choices(list(probabilities[current_char].keys()),
                                       weights=list(probabilities[current_char].values()))[0]
            result += next_char
            current_char = next_char
        else:
            break
    return result

def text_to_speech(text, lang='ja'):
    if not text:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except gTTSError as e:
        st.error(f"音声の生成中にエラーが発生しました: {str(e)}")
        return None

def autoplay_audio(file):
    if file is None:
        return
    audio_base64 = base64.b64encode(file.getvalue()).decode()
    audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}">'
    st.markdown(audio_tag, unsafe_allow_html=True)

st.title("マルコフ連鎖テキスト生成器")

# 入力テキスト
text = st.text_input("入力テキスト", value="しかのこのこのここしたんたん")

# マルコフモデルの作成
model = create_markov_model(text)

# 遷移確率の計算
probabilities = calculate_transition_probabilities(model)

if st.checkbox("遷移確率を表示"):
    st.subheader("マルコフモデルに基づく遷移確率:")
    for char, transitions in probabilities.items():
        st.write(f"遷移元: '{char}'")
        for next_char, prob in transitions.items():
            st.write(f"  → '{next_char}': {prob:.2f}")
        st.write()

start_char = st.text_input("開始文字", value="し")

# セッション状態の初期化
if 'running' not in st.session_state:
    st.session_state.running = False
if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ""
if 'audio' not in st.session_state:
    st.session_state.audio = None

# 開始/停止ボタン
if st.button("開始" if not st.session_state.running else "停止"):
    st.session_state.running = not st.session_state.running

# 生成と再生の間隔（秒）
interval = st.slider("生成間隔（秒）", 1, 10, 5)

# テキスト生成と音声再生の領域
text_area = st.empty()

# 自動生成と再生
if st.session_state.running:
    st.session_state.generated_text = generate_text(probabilities, start_char)
    text_area.write(f"生成されたテキスト: {st.session_state.generated_text}")
    
    audio_fp = text_to_speech(st.session_state.generated_text)
    if audio_fp:
        st.session_state.audio = audio_fp
        autoplay_audio(st.session_state.audio)
    else:
        st.warning("音声を生成できませんでした。テキストのみ表示します。")
    
    time.sleep(interval)
    st.experimental_rerun()

# 停止時も最後に生成されたテキストを表示
elif st.session_state.generated_text:
    text_area.write(f"最後に生成されたテキスト: {st.session_state.generated_text}")
```

## 解説

1. **ライブラリのインポート**: 必要なライブラリをインポートします。
2. **マルコフモデル関連関数**: `create_markov_model`, `calculate_transition_probabilities`, `generate_text` はマルコフモデルの構築、遷移確率の計算、テキスト生成を行う関数です。
3. **音声合成関連関数**: `text_to_speech`, `autoplay_audio` はテキストを音声に変換し、再生する関数です。
4. **Streamlit アプリケーション**: Streamlit を用いて、ユーザーインターフェースを構築します。
    * テキスト入力: `st.text_input` でユーザーがテキストを入力できます。
    * 遷移確率の表示: `st.checkbox` と `st.write` で遷移確率を表示するかどうかを選択できます。
    * 開始文字の入力: `st.text_input` でテキスト生成の開始文字を入力できます。
    * 開始/停止ボタン: `st.button` でテキスト生成と音声再生の開始/停止を制御します。
    * 生成間隔: `st.slider` でテキスト生成と音声再生の間隔を設定できます。
    * テキストと音声の出力: `st.empty` と `st.write`, `autoplay_audio` で生成されたテキストと音声を表示、再生します。

## まとめ

この記事では、「しかのこのこのここしたんたん」というフレーズをマルコフ連鎖モデルを用いて生成するデモを作成しました。マルコフモデルに基づいたテキスト生成と音声再生を体感することで、確率的なテキスト生成の面白さを実感できます。ぜひ、[デモサイト](https://shikanoko.streamlit.app)で実際にお試しください。