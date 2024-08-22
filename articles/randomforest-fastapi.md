---
title: "poetry+fastapi+render+streamlitを用いて機械学習で推論するapiを作成してみた"
published: true
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["render", "FastAPI", "Streamlit", "機械学習","RandomForest"]
emoji: "🌸"
---

## はじめに

APIを作成し、機械学習モデルをウェブアプリケーションから利用する方法を学びたいと思い、このプロジェクトを開始しました。RenderとStreamlitを使用することで、無料でデプロイできることを知り、有名なIrisデータセットを使用した機械学習モデルをAPIに組み込んだ[デモサイト](https://myrandomforestapi.streamlit.app/)を構築しました。ぜひ実際に体験してみてください。

https://myrandomforestapi.streamlit.app/

![](/images/fastapi.png)

## Irisデータセットについて

Irisデータセットは、機械学習の分野で広く使用される有名なデータセットです。このデータセットには、3種類のアヤメ（Iris setosa、Iris versicolor、Iris virginica）の花の測定データが含まれています。

各サンプルには以下の4つの特徴（入力データ）があります：
1. がく片の長さ（Sepal Length）：cm単位
2. がく片の幅（Sepal Width）：cm単位
3. 花弁の長さ（Petal Length）：cm単位
4. 花弁の幅（Petal Width）：cm単位

これらの特徴を基に、アヤメの品種（出力データ）を予測します。
本プロジェクトでは、Kaggleから取得したIrisデータセットを使用しています。

## プロジェクトの概要

このプロジェクトでは、FastAPIとStreamlitを使用して、Irisデータセットの品種を予測するアプリケーションを作成します。具体的には以下の要素で構成されています：

1. RandomForestClassifierを使用した機械学習モデル：4つの入力特徴からIrisの品種を予測
2. FastAPIを用いたバックエンドAPI：予測モデルをWeb経由で利用可能にする
3. Streamlitで構築したインタラクティブなフロントエンド：ユーザーが簡単に入力データを提供し、予測結果を視覚的に確認できる

モデルのトレーニングはローカル環境で行い、学習済みモデルをFastAPIで提供し、Streamlitで可視化します。

## プロジェクトの構成

プロジェクトは以下のようなディレクトリ構成で管理します：

```
fastapi_project/
├── main.py               # FastAPIのエントリーポイント
├── models/
│   └── iris_model.pkl    # 学習済みモデルファイル
├── pyproject.toml        # Poetryによる依存関係管理ファイル
├── poetry.lock           # Poetryによるロックファイル
├── data/     
│   └── iris.csv          # Kaggleから取得したIrisデータセット
└── model_training.py     # モデルのトレーニングスクリプト

streamlit_project/
├── app.py                # Streamlitアプリケーションのエントリーポイント
├── poetry.lock           # Poetryによる依存関係管理ファイル
└── pyproject.toml        # Poetryによるロックファイル
```

## 依存関係の管理

Poetryを使用して依存関係を管理します。`pyproject.toml`ファイルには以下の内容を記述します：

### FastAPI プロジェクト

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.112.1"
uvicorn = "^0.30.6"
jinja2 = "^3.1.4"
python-multipart = "^0.0.9"
pandas = "^2.2.2"
mangum = "^0.17.0"
numpy = "^2.1.0"
matplotlib = "^3.9.2"
seaborn = "^0.13.2"
scikit-learn = "^1.5.1"
joblib = "^1.4.2"
```

### Streamlit プロジェクト

```toml
[tool.poetry.dependencies]
python = "^3.12"
streamlit = "^1.37.1"
requests = "^2.32.3"
pandas = "^2.2.2"
```

## 機械学習モデルのトレーニング

`model_training.py`を使用してRandomForestClassifierモデルをトレーニングし、保存します：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Kaggleから取得したIrisデータセットの読み込みと前処理
df = pd.read_csv("iris.csv")
X = df.drop("species", axis=1)  # 4つの特徴量を入力として使用
y = df["species"]  # 品種を予測対象として使用

# データの分割（学習用と検証用）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestClassifierモデルのトレーニング
model = RandomForestClassifier()
model.fit(X_train, y_train)

# モデルの保存（後でFastAPIで使用）
joblib.dump(model, "models/iris_model.pkl")
```

## FastAPIバックエンドの実装

`main.py`にFastAPIのエンドポイントを設定します。このAPIは以下の機能を提供します：

1. Irisの4つの特徴量（がく片の長さ・幅、花弁の長さ・幅）を受け取り、品種を予測
2. 予測結果と各品種（setosa、versicolor、virginica）の予測確率を返す

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 学習済みモデルのロード
model = joblib.load("./models/iris_model.pkl")

# 入力データのスキーマを定義
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict/")
def predict(input_data: IrisInput):
    # 入力データを配列に変換
    data = np.array([[input_data.sepal_length, input_data.sepal_width,
                      input_data.petal_length, input_data.petal_width]])
    
    # 品種の予測
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    
    # 品種のラベル名リスト
    label_names = model.classes_.tolist()
    
    return {
        "prediction": prediction[0],
        "prediction_label": label_names[prediction[0]],
        "prediction_proba": prediction_proba[0].tolist()
    }
```

## Streamlitフロントエンドの実装

`streamlit_app.py`にStreamlitアプリケーションのコードを記述します。このアプリケーションでは以下の機能を提供します：

1. ユーザーがIrisの4つの特徴量を入力できるフォーム
2. 入力された特徴量をAPIに送信し、予測結果を取得
3. 予測された品種、各品種の予測確率の表示

```python
import streamlit as st
import requests

API_URL = "https://your-deployed-fastapi-url.onrender.com"

st.title("Iris Flower Species Prediction")

st.write("""
このアプリケーションでは、アヤメの花の特徴を入力すると、その品種を予測します。
以下の4つの特徴を入力してください：
""")

# 入力フォーム
st.header("アヤメの特徴を入力してください")

sepal_length = st.number_input("がく片の長さ (cm)", min_value=0.0, value=5.1, format="%.2f")
sepal_width = st.number_input("がく片の幅 (cm)", min_value=0.0, value=3.5, format="%.2f")
petal_length = st.number_input("花弁の長さ (cm)", min_value=0.0, value=1.4, format="%.2f")
petal_width = st.number_input("花弁の幅 (cm)", min_value=0.0, value=0.2, format

="%.2f")

# 予測ボタン
if st.button("予測"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    response = requests.post(f"{API_URL}/predict/", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        st.write(f"予測品種: {result['prediction_label']}")
        st.write("予測確率:")
        for label, prob in zip(result['prediction_proba'], model.classes_):
            st.write(f"{label}: {prob:.2%}")
    else:
        st.error("予測に失敗しました。")
```

## デプロイ

- **FastAPIバックエンド**: Renderにデプロイ
- **Streamlitフロントエンド**: streamlit.ioにデプロイ


## まとめ

本プロジェクトでは、FastAPIとStreamlitを組み合わせて、Irisデータセットを用いた品種予測アプリケーションを作成しました。これにより、機械学習モデルを簡単にウェブアプリケーションで利用できるようになります。是非、このアプローチを応用して、他のデータセットやモデルにも挑戦してみてください。

---

