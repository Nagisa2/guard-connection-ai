# App_monitor 引き継ぎ資料

- 作成日: 2026-08-21
- 対象: プロジェクト引き継ぎ用
- 状態: 実装・検証継続中 / 主要機能は動作確認済み

---

## 1. プロジェクト概要

本アプリケーションは、医療現場向けの心電図・脈波モニタリングダッシュボード。
バックエンド側では FastAPI を用いて患者管理、認証、WebSocket 受信、アラート生成を担当し、
フロントエンド側では React + Vite で患者一覧、リアルタイムグラフ、履歴・プロフィール表示を提供している。

本アプリは現在、以下のような用途を想定している。

- 医師が患者一覧を確認する
- リアルタイムで ECG / PPG / HR を監視する
- 重大イベント時にアラートを確認する
- ホルター心電図の要約を参照する
- 将来的に JSON からの長時間データや自動要約機能を拡張する

---

## 2. 現在の実装状態

### 2.1 主要機能の現状

- 実装済み: 医師ログイン機能、患者一覧表示、患者登録、WebSocket でのリアルタイム表示、アラート一覧と既読管理、JSON 形式のデータ送信
- 実証中: Holter / 24h summary API は UI と API の骨組みが実装されているが、実運用レベルの診断ロジックまでは含まれていない

### 2.2 システム上の基本方針

- バックエンドとフロントエンドは分離した構成
- センサーデータは WebSocket を中心に流す
- フロントエンドでは `graphDataMap` に時系列データを保持して描画
- 患者一覧は API から取得し、選んだ患者に対してセンサーデータを紐付ける
- 存在しない患者データを避けるため、起動時に初期シードを投入する

---

## 3. 技術スタック

### バックエンド

- Python
- FastAPI
- SQLAlchemy
- SQLite
- JWT ベース認証
- WebSocket

### フロントエンド

- React
- Vite
- axios
- Recharts

### 主要依存関係

- `backend/main.py`
- `backend/models.py`
- `backend/crud.py`
- `backend/schemas.py`
- `backend/auth.py`
- `backend/test_sender.py`
- `frontend/src/Dashboard.jsx`
- `frontend/src/LoginPage.jsx`

---

## 4. フォルダ構成と役割

### ルート

- `心電図_打合せ1メモ.txt`: 要件・打合せメモ
- `打合せまとめ.docx`: 打合せの要約資料
- `ch2-01.png`: 1拍表示の参照イメージ
- `json_output/Data/`: ECG / PPG の JSON データ群

### backend/

- `main.py`
  - FastAPI アプリ本体
  - 認証、患者管理、WebSocket、Holter summary API を持つ
- `models.py`
  - Doctor / Patient / SensorRecord / Alert の DB 定義
- `crud.py`
  - DB 操作の実装
- `schemas.py`
  - API の request / response 定義
- `auth.py`
  - JWT 生成、検証
- `database/`
  - SQLite DB 保存先
- `test_sender.py`
  - JSON を読み込んで WebSocket へ送信する送信側スクリプト

### frontend/

- `src/App.jsx`: 認証状態の管理
- `src/LoginPage.jsx`: ログイン画面
- `src/Dashboard.jsx`: メイン画面、患者一覧、モニタリンググラフ、通知などの中心
- `src/HistoryTab.jsx`: 履歴確認
- `src/ProfileTab.jsx`: 患者プロフィール

---

## 5. 実行方法

### 5.1 バックエンド起動

```powershell
cd C:\path\to\App_monitor-v2
.\scripts\setup_demo_environment.ps1
.\scripts\start_backend.ps1
```

`uvicorn` を直接実行すると、移動前の仮想環境パスが埋め込まれた実行ファイルを呼ぶ場合がある。上記スクリプトは隔離環境のPythonから `python -m uvicorn` として起動するため、activate状態には依存しない。

### 5.2 データ送信スクリプト起動

```powershell
cd C:\path\to\App_monitor-v2\backend
.\.demo-venv\Scripts\python.exe test_sender.py
```

### 5.3 フロントエンド起動

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL = "http://localhost:8000"
npm run dev -- --host 0.0.0.0
```

ブラウザで以下へアクセス:

- http://localhost:5173

---

## 6. 認証と初期データ

### テスト用アカウント

- `yamada` / `password123`
- `suzuki` / `password123`

### 初期患者データ

起動時に以下の患者がシードされる。

- test_user_01: 田中 太郎
- test_user_02: 佐藤 花子
- test_user_03: 鈴木 一郎
- test_user_04: 山本 涼
- test_user_05: 井上 美咲
- test_user_06: 中村 健太
- test_user_07: 渡辺 直子
- test_user_08: 小林 勇太

### 注意事項

- データ送信側は患者を自動作成しない
- 事前にバックエンド起動時に患者一覧が存在している必要がある
- `user_id` と JSON パターンの対応が前提となる

---

## 7. データフローの整理

### 7.1 送信元

`backend/test_sender.py` が `json_output/Data` 配下の JSON を読み込んで、
以下の形式で WebSocket に送信する。

- `ECG`
- `PPG`
- `HR`

### 7.2 中継

バックエンドの `ws/sensors/{user_id}` でフロントエンド接続を受け、
`ws/device/signal/{user_id}` からのデータをクライアントへブロードキャストする。

### 7.3 表示

フロントエンドでは `Dashboard.jsx` の `graphDataMap` に値を保存し、
`MonitorChart` で時系列グラフを描画している。

### 7.4 実装上の注意

- バッチデータは 1 回ごとに複数点をまとめて送る設計
- `values` 配列の最後の値が最新値として扱われる
- これを使ってアラートや最新値カードの表示を更新している

---

## 8. `Holter` / summary 実装の現状

### 実装済みの内容

- `main.py` に `_build_holter_report()` を作成
- `GET /holter/{patient_id}` の API を追加
- `GET /summary/{patient_id}` の API を追加
- AF の疑いレベル、RR 変動、イベント数を仮算出するロジックを実装

### ここまでの設計意図

- 長時間心電図の概要表示をダッシュボード上に出すための仮の構造として扱う
- 実データに対する診断ロジックまではまだ作成していない
- 今は「UI と API の骨組み」を先に作る形で実装している

### 重要な留意点

- この処理は医療診断用の本格ロジックではなく、診断支援のためのサンプル実装である
- 実際の臨床判定としては、検証・評価と専門家レビューが必要である
- `AF likelihood` や `episode_count` はサンプルデータと仮条件に依存しており、臨床判断に直接使用しないこと
- 実運用時には、監査可能性や説明可能性を持つ処理設計を追加する必要がある

---

## 9. 課題と改善ポイント

### 9.1 課題

1. 現在の時系列グラフは 1 拍の特徴が見えづらい
   - 画面上では長い時間軸が主になっており、1 拍ごとの波形が平坦に見える
   - `ch2-01.png` のイメージに近づけるには、1 拍表示のデフォルト化やズームの強化が必要

2. 長時間データの保存設計が簡易的
   - `SensorRecord` は 1 レコード 1 時点を追加する設計になっている
   - 本格的な Holter 用データ保存では、時系列データの圧縮や分割が必要

3. AF 判定ロジックが仮の値計算
   - 現状はヒューリスティックな条件に基づく
   - 実運用前には専門家レビュー・検証が必要

4. 認証がテスト用ハードコード
   - 本番運用を考えるなら DB 管理、パスワードハッシュ化、ユーザー管理が必要

5. CORS / API URL の扱いが固定前提
   - `localhost` 前提の運用が強く、環境ごとの差異に弱い

### 9.2 今後の優先課題

#### 優先度A: すぐ必要

- 1拍表示の見せ方の最適化
- 長時間データとリアルタイムデータの切り分け
- 実データ読み込み対応の整理
- 初期表示の UX 改善（診療支援のための視認性向上）

#### 優先度B: 次にやるべきこと

- Holter summary の UI 強化
- AF 疑いイベントの履歴表示
- データ保存形式の整理
- ログや監査機能の導入

#### 優先度C: 将来拡張

- 本番向けの認証基盤導入
- ML / signal processing による自動判定
- 画像・PDF 報告書の自動生成

---

## 10. 設計判断とメモ

### 10.1 1拍表示に関する判断

- 画面の初期表示は時系列を中心にし、1拍はオプション表示にする方が実務上見やすい
- `ECG (1拍)` と `PPG (1拍)` は訓練・観察用途で使うと有効
- ただし、現時点ではグラフの大きさ・軸・ズーム設計の最適化が必要

### 10.2 Holter summary の位置づけ

- 現在の実装は「要約の仕組みの実証」段階
- まだ医療用レポートの完成品ではない
- 継続的に改善していく対象

---

## 11. 既知の動作・運用メモ

- バックエンドを起動していないと WebSocket 接続や患者データ取得が失敗する
- `uvicorn` の 8000 番ポートが既に占有されていると起動失敗するので要確認
- フロントの接続先が `ws://localhost:8000` 前提で作られている
- `json_output/Data` のデータがある場合に限り、テスト送信が成立する
- 初期状態で患者が存在しない場合は表示されないため、起動時のシードが重要

---

## 12. 引き継ぎメモ

- 基本機能は一旦動く状態のため、UI/UX と Holter 要約ロジックの整備を主に行っている
- 今後の改善として、診療支援としての見せ方とデータの本格利用可能性を重視する予定

---

## 13. 今後の実装予定

- グラフの UX 改善
- 時系列と 1拍を切り替える際の視認性の最適化
- Holter の過去データと現在データを分離した設計
- 医療向けの説明ができるレポート生成の検討

- 実データのインポート仕様の設計

---

## 14. PPG-only AF・疑似ECGリアルタイム統合

### 14.1 位置づけ

疑似ECGはPPG pulse intervalの補助表示であり、実測ECG、診断用ECG、確定診断ではない。画面では常に「PPG由来疑似ECG・診断用ではない」と表示し、P波、PR、QRS幅、QT、STの測定には使用しない。

### 14.2 API

- `POST /realtime/sessions`: PPG処理sessionを作成
- `DELETE /realtime/sessions/{session_id}`: sessionを終了
- `POST /realtime/sessions/{session_id}/chunks`: HTTP POSTでPPG chunkを処理
- `WS /ws/realtime/{session_id}`: WebSocketでPPG chunkを連続処理
- `WS /ws/sensors/{user_id}`: 処理済み波形を医師画面へ配信

リアルタイム運用ではWebSocketを推奨する。HTTP POSTはデバッグ、単発試験、再送制御用である。

### 14.3 起動環境

`guard-connection-ai` をimportでき、NumPy/SciPyを含むPython環境が必要である。ローカルの並列配置では自動的に `../guard-connection-ai/src` を検出する。別配置では `GUARD_AI_SRC` に同リポジトリの `src` を指定する。

```powershell
cd backend
python -m pip install -r requirements.txt
$env:GUARD_AI_SRC = "C:\path\to\guard-connection-ai\src"
uvicorn main:app --host 0.0.0.0 --port 8000
```

既存の `test_sender.py` が送るECGは実測ではなく表示試験用であるため、`waveform_type=simulated_ecg`、`diagnostic_ecg=false` を付与している。

### 14.4 Deployment profile

session作成時は `deployment_profile_id` を指定する。scaler、PAT、SQI設定はサーバー側profileから取得し、デバイスからの上書きを受け付けない。

```json
{
  "schema_version": "1.0",
  "user_id": "test_user_01",
  "sample_rate_hz": 125,
  "deployment_profile_id": "bidmc_train43_ppg_v1"
}
```

`bidmc_train43_ppg_v1` はBIDMCのtrain patient 43名だけから生成した前処理研究用artifactであり、臨床検証済みではない。artifact hash、入力ファイルhash、split hashを保持する。AFモデル未接続中は `af_probability` と `uncertainty` はnullであり、chunk送信側からの注入は拒否する。
- Holter summary の JSON スキーマ整理
- 患者ごとに異なる測定パターンの導入
- 自動アラート判定ロジックの構築

---

## 15. コンテスト用デモ経路

### 15.1 今回追加した範囲

他担当の実機・後段AIを待たずに一連の画面遷移を検証できるよう、次を追加した。

- 記録済み実PPGを通常のリアルタイムAPIへ等速再生する送信機
- `live_device`、`recorded_demo_replay`、`synthetic_demo`を区別する入力出所
- AF疑い通知、患者応答、医師確認、家族確認、緊急化、解決の状態遷移
- AF確率を「診断結果」ではなく「後段AIによるAF疑い」として表示する医師画面
- デモ記録の常時表示

デモ用AF記録の正解ラベルは再生ファイルの選択にだけ使い、推論APIへ入力しない。これにより、正解ラベルを推論結果として流用する経路を作らない。

### 15.2 記録済みPPGの再生

バックエンド起動後、別のPowerShellで次を実行する。

```powershell
cd <workspace>\App-monitor-v2\backend
python recorded_ppg_demo_sender.py
```

既定では `guard-connection-ai/artifacts/team_handoff/downstream_ai_v1_1/af_good.jsonl` の元PPGとvalid maskだけを読み、`test_user_02` の記録済みAFシナリオとして等速再生する。疑似ECG、実測ECG、正解ラベルは送信しない。

主要なオプションは次のとおり。

```powershell
python recorded_ppg_demo_sender.py --help
python recorded_ppg_demo_sender.py --loop
python recorded_ppg_demo_sender.py --user-id test_user_02 --scenario-id af_recorded_01
```

停止は `Ctrl+C` を使用する。スタックトレースを出さず、作成したセッションを終了して停止する。

### 15.2.1 一括起動

初回だけ、移動で壊れた既存の `venv` を変更せず、隔離したデモ環境を作成する。

```powershell
cd <workspace>\App-monitor-v2
.\scripts\setup_demo_environment.ps1
```

以降は次の1コマンドで、バックエンド、医師画面、正常例とAF記録例の同期再生、AF疑い、患者の体調不良応答までを起動できる。

```powershell
.\scripts\start_contest_demo.ps1
```

ブラウザからシナリオを選んで開始・停止する場合は、次を実行する。

```powershell
cd C:\path\to\App_monitor-v2
.\scripts\start_contest_demo.ps1 -ControlPanel
```

`http://127.0.0.1:5173/?mode=control` が開き、「本人回答待ち」「体調不良」「救助要請」の3シナリオを操作できる。制御画面の停止はシナリオだけを停止し、PowerShell側の `Ctrl+C` はバックエンドとフロントエンドも含めて終了する。

この起動方法では、医師画面の`test_user_01`と`test_user_02`に表示専用のシミュレーションECGも同時配信する。コントロールパネルでシナリオを開始すると、PPGとPPG由来疑似ECGは前段AI経路の信号へ切り替わる。シミュレーションECGは表示確認用の独立したテンプレートで、同時表示するPPGに対応する実測ECGではなく、診断や波形再現精度の比較には使用できない。

制御画面のAFデモは、後段AI受信APIへ`demo_stub`として候補状態を送り、確認状態へ遷移した場合だけcare eventを作成する。画面では実モデル判定と区別して表示する。
`backend/test_sender.py`と制御画面は同じ患者IDへ別の波形を送るため、通常は同時に起動しない。誤って同時起動した場合、医師画面はリアルタイムセッションを優先し、異なる送信元の波形を同じグラフへ混在させない。

シミュレーションECGだけを手動で補助配信する場合は次を使用する。

```powershell
cd backend
.\.demo-venv\Scripts\python.exe test_sender.py --user-ids test_user_01 test_user_02 --signal-types ECG
```

医師画面は `http://127.0.0.1:5173`、デモ用ログインは `yamada / password123` である。終了時は、このスクリプトが起動したプロセスだけをPIDで停止する。

同じフロントエンドから、患者・家族の接続試験用画面も開ける。

- 患者: `http://127.0.0.1:5173/?mode=patient&user_id=test_user_02`
- 家族: `http://127.0.0.1:5173/?mode=family&user_id=test_user_02`

これらは実際の患者・家族アプリではなく、APIと状態遷移を単独検証するローカルデモ画面である。実際の緊急通報は行わない。

患者応答は引数で切り替えられる。

```powershell
.\scripts\start_contest_demo.ps1 -PatientAction patient_ok
.\scripts\start_contest_demo.ps1 -PatientAction patient_unwell
.\scripts\start_contest_demo.ps1 -PatientAction patient_help
.\scripts\start_contest_demo.ps1 -PatientAction none
```

`-Loop` は停止要求まで波形を繰り返す。`-NoWait` は結合試験専用であり、プレゼンでは使用しない。

既定の記録済みサンプルが存在しない公開用コピーでは、合成した不規則脈波へ自動的に切り替わる。この場合は `source_mode=synthetic_demo` と表示し、実患者のAF記録とは扱わない。

### 15.3 入力出所

セッションと入力packetには、次の組み合わせを必須とする。

| 利用形態 | `source_mode` | `demo_scenario_id` |
|---|---|---|
| 実機 | `live_device` | 指定不可 |
| 記録済みデモ | `recorded_demo_replay` | 必須 |
| 合成デモ | `synthetic_demo` | 必須 |

セッションとpacketで値が一致しない場合は `source_provenance_mismatch` として拒否する。同一セッションを実機入力からデモ入力へ切り替えない。

### 15.4 AF疑いイベントAPI

- `POST /care-events/af-suspicions`: 後段AIのAF疑いを登録
- `POST /care-events/{event_id}/actions`: 患者・家族・医師の操作を登録
- `GET /care-events`: イベント一覧
- `GET /care-events/{event_id}`: 個別イベント
- `WS /ws/sensors/{user_id}`: `care_event` を医師画面へ配信

AF疑いだけで自動的に緊急通報へ遷移しない。患者の救助要請、応答なし、転倒検知、または医師の操作で緊急状態へ進む。イベント、状態、作成・操作の冪等性キーは `backend/database/care_events.db` へ保存し、バックエンド再起動後に復元する。

### 15.5 現時点の制約

- 後段AFモデルとのイベント自動接続は未実装。AF疑いAPIは統合契約の先行実装である
- 一括デモのAF疑いは `inference_mode=demo_stub` であり、実モデル判定と型・画面表示の両方で区別する
- 実際の患者・家族アプリからの応答送信は未接続。ローカルデモ画面のみ実装済み
- 固定デモ位置は大島商船高等専門学校（33.938502, 132.190863）とし、医師・患者・家族画面へ同期する。地図はLeafletとOpenStreetMap、道路経路はOSRMを使用する。出発地点は大畠駅付近に置いたデモ救急車待機地点であり、実際の消防署・救急車位置ではない
- OSRMに接続できない場合は直線経路へ自動的に切り替える。実機GPS、交通状況、救急通報は未接続
- 地図と道路経路は座標を初めて受信したときに初期化する。同じ座標を1秒ポーリングで再受信しても再初期化せず、利用者が行ったズーム・移動操作を維持する
- 緊急通報は外部へ送信せず、デモ内の状態遷移に限定する
- `_build_holter_report()` の値はUI骨組み用の仮値であり、後段AIの結果ではない
- バックエンドの既存仮想環境は移動前のPythonパスを参照しているため、再作成が必要である
