---
title: "マルコフ連鎖を用いたインタラクティブなテキスト生成アプリの作成"
slug: "markov-chain-text-generator"
published: true
type: tech
topics:
  ["Python", Streamlit
  - 自然言語処理
  - マルコフ連鎖
emoji: 🦚
---

# 『しかのこのこのここしたんたん』をマルコフモデルで考えてみた

本記事では、Streamlitを用いて、入力されたテキストからマルコフ連鎖に基づいて新しいテキストをリアルタイムに生成し、さらにそれを音声で再生するウェブアプリケーションを作成する方法を紹介します。
実際に動作するデモを[こちら](https://shikanoko.streamlit.app)で公開しているので、是非試してみてください。

![](/images/shikanoko_app.png)

### はじめに

マルコフ連鎖は、過去の状態のみに基づいて未来の状態が確率的に決定される確率モデルです。自然言語処理の分野では、文章生成などに活用されています。

本記事では、このマルコフ連鎖を用いて、入力されたテキストデータから学習し、新しいテキストを自動生成するアプリケーションを開発します。さらに、生成されたテキストを音声で再生することで、よりインタラクティブな体験を提供します。

### 「しかのこのこのここしたんたん」を例にした説明

例えば、「しかのこのこのここしたんたん」というテキストを入力した場合、アプリケーションは以下のようにマルコフモデルを構築します。

**マルコフモデルに基づく遷移確率:**

* **遷移元: 'し'**
    * → 'か': 0.50 (「しか」と出現する確率が50%)
    * → 'た': 0.50 (「した」と出現する確率が50%)
* **遷移元: 'か'**
    * → 'の': 1.00 (「かの」と必ず出現)
* **遷移元: 'の'**
    * → 'こ': 1.00 (「のこ」と必ず出現)
* **遷移元: 'こ'**
    * → 'の': 0.50 (「この」と出現する確率が50%)
    * → 'こ': 0.25 (「ここ」と出現する確率が25%)
    * → 'し': 0.25 (「こし」と出現する確率が25%)
* **遷移元: 'た'**
    * → 'ん': 1.00 (「たん」と必ず出現)
* **遷移元: 'ん'**
    * → 'た': 1.00 (「んた」と必ず出現)

このマルコフモデルに基づいて、例えば「し」から始まるテキストを生成する場合、最初は「し」の次に「か」が来る確率と「た」が来る確率がそれぞれ50%となります。もし「か」が選ばれた場合、次は必ず「の」が来ます。このようにして、確率的に文字をつなげていくことで、"しかのこたんたん" や "したんこのこのこ" のような、入力テキストの傾向を持った新たなテキストが生成されるのです。 

### アプリケーションの概要

本アプリケーションは、以下の機能を備えています。

* テキスト入力：ユーザーが任意のテキストを入力できます。
* マルコフモデルの構築：入力されたテキストデータに基づいて、マルコフモデルを構築します。
* テキスト生成：構築したマルコフモデルに基づいて、新しいテキストを自動生成します。
* 音声再生：生成されたテキストを音声合成エンジンを用いて音声に変換し、再生します。
* リアルタイム処理：テキスト生成と音声再生をリアルタイムに行います。

### 開発環境

* Python 3.7 以上
* Streamlit
* gTTS
* collections

### ソースコード

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

### 解説

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



### まとめ

本記事では、Streamlitを用いてマルコフ連鎖に基づくリアルタイム音声付きテキスト生成器を作成する方法を紹介しました。このアプリケーションは、ユーザーが自由にテキストを入力し、マルコフモデルに基づいて生成されるユニークなテキストと音声を体験することができます。
さらに、[デモサイト](https://shikanoko.streamlit.app)で実際に動作を確認することで、より理解を深めることができます。
