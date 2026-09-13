# GUARD Connection

PPG入力から非診断用の疑似ECGを生成し、医師向け画面へリアルタイム配信するデモ用モノレポです。

## 構成

- `guard-connection-ai`: PPG前処理、Timing Head、形態prior、ストリーミング生成
- `App-monitor-v2`: FastAPIバックエンド、React医師向け画面、デモ送信機

疑似ECGは`ppg_derived_pseudo_ecg`として扱い、実測ECGや診断用ECGとは区別します。P波、PQ/PR、QT、ST、患者固有QRS形態を正確に復元したものではありません。AF判定モデルはこの公開物に含まれません。

## デモ起動

初回だけ依存関係を準備します。

```powershell
.\App-monitor-v2\scripts\setup_demo_environment.ps1
```

以降は1コマンドでバックエンド、医師画面、2患者の波形、デモAF疑い、患者応答を起動します。

```powershell
.\App-monitor-v2\scripts\start_contest_demo.ps1
```

ブラウザで`http://127.0.0.1:5173`を開きます。デモ用ログインは`yamada / password123`です。本番認証ではありません。別添の記録済みPPGがない環境では合成不規則脈波へ切り替わり、`synthetic_demo`として表示されます。デモAF疑いは`demo_stub`であり、後段AIの実モデル判定ではありません。

## データについて

学習・評価データ、個人情報、MLflow記録、開発用DBは含みません。3個の小容量モデルはデモ再現のために同梱しています。モデル出力は臨床検証済みではありません。

## 担当者向け仕様

- `guard-connection-ai/docs/HANDOFF_DEVICE_PATIENT_APP.md`
- `guard-connection-ai/docs/HANDOFF_DOWNSTREAM_AI.md`

各担当者向けJSONLサンプルはGitHubには含めず、別添ZIPで共有します。
