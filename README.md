# 奶龙奶蛙识别机器人

这是一个基于 MobileNetV3-Small 的三分类 CNN 项目，可识别奶龙、奶蛙和其他图片，并通过飞书机器人回复结果。

## 数据目录

```text
data/
├── train/
│   ├── nailong/
│   ├── naiwa/
│   └── other/
└── test/
    ├── nailong/
    ├── naiwa/
    └── other/
```

当前数据只有 38 个训练样本和 9 张测试图，适合验证完整链路，还不足以证明模型已经具备稳定的线上泛化能力。训练时会对 GIF 随机抽帧，并提高动画硬负样本的采样权重，减少模型只按颜色和轮廓判断。后续仍需继续补充复杂背景、截图压缩、局部遮挡和群聊常见图片。

## 本地开发

项目使用 Python 3.11 和 `uv`：

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

训练并评估：

```bash
uv run nailong-train
```

产物保存在：

```text
artifacts/model.pt
artifacts/metrics.json
```

测试单张图片：

```bash
uv run nailong-predict data/test/nailong/image.png
```

## 飞书机器人

在飞书开放平台创建企业自建应用，开启机器人能力并完成以下配置：

- 使用 WebSocket 长连接订阅事件。
- 订阅接收消息事件。
- 申请接收消息、下载消息资源和机器人回复消息所需权限。
- 猜图抢答要显示真名，需额外开通通讯录读取权限 `contact:user.base:readonly`；没开通时会退回“神秘奶友”，不影响积分。
- 权限变更后重新发布或安装应用。

本地启动：

```bash
export LARK_APP_ID=cli_xxx
export LARK_APP_SECRET=xxx
uv run nailong-bot
```

机器人会自动处理群里的普通图片和表情：

- 直接发送图片或表情，不用 `@`。识别到奶龙会随机回复一张奶龙图，识别到奶蛙会随机回复一张奶蛙图。
- 接龙会优先发送本群尚未用过的收集图，再轮换种子图；同类图片全部轮过后才会重复。
- 需要查看文字识别结果时，回复原图片并 `@` 机器人。`@` 识别时除了识别结论，还会附上「奶龙浓度检测仪」读数（把奶龙和奶蛙的概率之和当作奶含量），奶龙/奶蛙图再附一行「稀有度」评级。
- 稀有度按这张图在群里被发送的累计次数评定：发得越少越稀有，只冒过一两次是传说，天天刷屏就是烂大街。
- 奶龙或奶蛙置信度达到 65% 时接龙并收进图鉴，低于 65% 时群聊静默。
- 单聊中遇到“其他”或低置信结果时也会返回文字提示；群聊保持静默，避免刷屏。
- 没见过的图片按 SHA-256 去重后收入图库，同一张图重复发送不会重复收录。
- 回复原图或机器人的接龙图，发送 `纠正为奶龙`、`纠正为奶蛙` 或 `纠正为其他`，可以修正分类。
- 每天北京时间 23:55，机器人会向当天出现过图片或表情的群自动发送“今日含奶量”。发送 `今日含奶量` 也可以随时手动查看。
- 发送 `奶龙图鉴`，查看群聊最近收集的 9 张奶龙或奶蛙图片。只有模型通过置信度门槛，或经人工明确纠正为奶龙/奶蛙的图片才进入图鉴；“其他”和不确定图片不进入图鉴。图鉴不计算接龙种子，每张图片都可以单独查看。
- 配置图鉴文档后，新图片首次进入图鉴时会自动追加到目标章节的最后一个四列网格。重复图片、最高候选为“其他”的图片不会追加；同步失败会保留进度并自动重试。
- 发送 `猜图`（或 `盲盒猜图`）开一局盲盒猜图：机器人从图鉴里随机抽一张奶图，裁出局部并打上高斯模糊出题，群友直接发 `奶龙` 或 `奶蛙` 抢答，本群第一个答对的人得 1 分并亮出完整原图。发 `公布答案` 揭晓本局但不给任何人加分，发 `猜图榜` 查看本群积分排行。积分按群聊独立统计。
- 发送 `@我是奶龙 help` 或 `@我是奶龙 帮助`，查看完整玩法。

普通文字会被忽略。每条消息只处理第一张图，超过 20 MB 的图片不会进入识别。

定时播报时间可通过 `DAILY_REPORT_TIME=HH:MM` 调整。机器人每分钟轮询一次，只要当天已过播报点且还没发过就补发；同一个群同一天只会播报一次，服务重启或错过时间点都会在下一次轮询自动补上，不会重复发送。

接龙图库和统计数据保存在 `BOT_STATE_DIR`。本地默认使用 `runtime/`，服务器使用 `/var/lib/nailong-bot/`。

自动追加图鉴文档需要配置：

```bash
CATALOG_DOCUMENT_ID=目标新版文档的 document_id
CATALOG_SECTION_BLOCK_ID=目标章节标题的 block_id
CATALOG_BASELINE_ASSET_IDS=首次启用前已经人工导入的图鉴编号，以逗号分隔
CATALOG_SYNC_RETRY_SECONDS=60
```

机器人应用需要拥有目标文档的编辑权限，并开通新版文档编辑、上传云文档图片所需权限。`CATALOG_BASELINE_ASSET_IDS` 只用于首次启用时登记已有图片；之后每张新图会按数据库同步状态去重，不需要继续修改该配置。

首次启动前，可把训练集导入为接龙种子图库：

```bash
uv run nailong-seed --state-dir runtime --data-dir data/train
```

## 部署到 tx

部署不使用 Docker。本地训练完成后执行：

```bash
./scripts/deploy.sh tx
```

脚本会把本地代码和模型同步到 `/opt/nailong-bot`，在服务器创建 Python venv，安装 CPU 版 PyTorch，并配置 `nailong-bot.service`。运行数据由 systemd 放在 `/var/lib/nailong-bot`，以后重新部署不会清空图库和统计。

首次启动前，在服务器创建凭据文件：

```bash
ssh tx
cp /opt/nailong-bot/deploy/nailong-bot.env.example /etc/nailong-bot.env
chmod 640 /etc/nailong-bot.env
```

填入真实的 `LARK_APP_ID` 和 `LARK_APP_SECRET` 后启动：

```bash
systemctl enable --now nailong-bot
systemctl status nailong-bot
journalctl -u nailong-bot -f
```

凭据文件位于 `/etc`，不会同步回本地，也不会进入 Git。

## Git 约定

本地目录是唯一源码工作区。服务器只保存运行副本，不直接修改代码。训练图片、模型权重、虚拟环境和凭据均已加入 `.gitignore`。
