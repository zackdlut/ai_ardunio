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

可选：将自训 YOLO 权重放到 `models/best.pt`。未放置时使用 `yolov8n.pt`（COCO）占位，仅用于跑通管线。`YOLO_DEVICE=auto`（默认）时，若本机有 Intel NPU，检测会经 OpenVINO 跑在 NPU 上；没有 NPU 或编译失败时仍用 CPU。

## 运行

```bash
python src/main.py
```

## 功能概览

- 摄像头预览 + YOLO 检测框
- 项目步骤（Blink）与 OpenCV 叠加提示
- Chat 调用内网 Ollama（`OPENAI_BASE_URL` / `OPENAI_MODEL`）
- 手势：拇指赞=下一步，握拳=暂停，V 字=重新识别
- 「烧录 Blink」按钮 / Agent 工具 `compile_and_upload`
- 语音：按住说话（本地 Whisper 或远程 Ollama ASR）+ 可选「语音回复」（本机 TTS）
- 导师模式：多轮对话、步骤切换朗读、视觉不匹配时主动提醒（见 `docs/ARCHITECTURE.md`）

## LLM 配置

见 `.env`：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 及 `NO_PROXY=10.67.34.44`。

## 语音配置

| 变量 | 说明 |
|------|------|
| `ASR_BACKEND` | `local` / `remote` |
| `OLLAMA_ASR_MODEL` | 远程 ASR 模型名 |
| `TTS_ENABLED` | 是否默认开启语音回复 |
| `TTS_VOICE` | Windows 音色名（如 `Microsoft Huihui Desktop`） |
| `TTS_RATE` | 朗读语速 |

侧栏可切换 ASR 后端、开关「语音回复」；按住说话时再次录音会打断朗读。
