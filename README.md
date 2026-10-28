# Arduino AI Learning Copilot (MVP)

桌面端 Arduino 实验助教：摄像头视觉、步骤引导、Arduino CLI 烧录、Ollama（OpenAI 兼容 API）对话。

## 环境

- Python 3.10+
- [Arduino CLI](https://arduino.github.io/arduino-cli/)（`arduino-cli core install arduino:avr`）
- 复制 `.env.example` 为 `.env`（仓库已可自带 `.env`）

## 安装

```bash
cd c:\media\workspace\ai_ardunio
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

若访问 `pypi.org` 超时，可使用镜像（示例）：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

可选：将自训 YOLO 权重放到 `models/best.pt`。未放置时使用 `yolov8n.pt`（COCO）占位，仅用于跑通管线。`YOLO_DEVICE=gpu`（默认）时，检测跑在 Intel 核显上。也可设为 `npu`、`cpu` 或 `auto`（优先 NPU）。

## 运行

```bash
python src/main.py
```

## 功能概览

- 摄像头预览 + YOLO 检测框
- 项目步骤（Blink）与 OpenCV 叠加提示
- Chat 调用任意 OpenAI 兼容服务（默认内网 Ollama，`OPENAI_BASE_URL` / `OPENAI_MODEL`）
- 手势：拇指赞=下一步，握拳=暂停，V 字=重新识别
- 「烧录 Blink」按钮 / Agent 工具 `compile_and_upload`
- 语音：按住说话（本地 Whisper、远程 Ollama ASR 或 OpenAI 兼容 ASR API）+ 可选「语音回复」（本机 TTS）
- 导师模式：多轮对话、步骤切换朗读、视觉不匹配时主动提醒（见 `docs/ARCHITECTURE.md`）

## LLM 配置

Chat 走 OpenAI 兼容的 `/chat/completions`（需支持 function calling 与流式），可以是内网 Ollama / vLLM，也可以是 DeepSeek、DashScope、OpenAI 等云端服务。

| 变量 | 说明 |
|------|------|
| `OPENAI_BASE_URL` | 服务根地址（含 `/v1`），如 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `OPENAI_API_KEY` | API Key；Ollama 随便填 |
| `OPENAI_MODEL` | 模型 id |
| `LLM_VISION` | `auto`（默认）/ `true` / `false`：是否给模型发摄像头画面 |
| `LLM_EXTRA_BODY` | 附加请求体 JSON，如 `{"enable_thinking": false}` |
| `LLM_TIMEOUT_S` | 单次请求超时，默认 600 秒 |

网络：内网主机要写进 `NO_PROXY`（直连）；云端服务经 `HTTPS_PROXY` 出网。服务不提供 `/models` 时，启动检查会跳过模型名校验。

也可以在应用顶栏点「设置」修改对话模型与语音识别的配置。保存时会先校验，再写回 `.env`，不用重启：改了对话模型会自动重连，改了 ASR 会立即切换。

## 语音配置

| 变量 | 说明 |
|------|------|
| `ASR_BACKEND` | `local` / `ollama`（旧值 `remote` 等同）/ `openai` |
| `ASR_LANGUAGE` | 识别语言，默认 `zh`，留空自动检测 |
| `WHISPER_MODEL_SIZE` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` | 本地 faster-whisper 的模型名或目录、设备、精度 |
| `OLLAMA_ASR_MODEL` | Ollama ASR 模型名 |
| `ASR_BASE_URL` / `ASR_API_KEY` | OpenAI 兼容 ASR 地址与 Key，留空沿用 `OPENAI_*` |
| `ASR_MODEL` | OpenAI 兼容 ASR 模型，如 `whisper-1`、`FunAudioLLM/SenseVoiceSmall`、`qwen3-asr-flash` |
| `ASR_API_MODE` | `transcriptions`（`/audio/transcriptions`）或 `chat`（`/chat/completions` + `input_audio`） |
| `ASR_EXTRA_BODY` | 附加请求体 JSON，如 `{"asr_options": {"language": "zh"}}` |
| `TTS_ENABLED` | 是否默认开启语音回复 |
| `TTS_VOICE` | Windows 音色名（如 `Microsoft Huihui Desktop`） |
| `TTS_RATE` | 朗读语速 |

语音菜单可在三种 ASR 后端间切换、开关「语音回复」；按住说话时再次录音会打断朗读。
