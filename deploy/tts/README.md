# 现有飞书和 QQ 机器人调用开发机语音服务

飞书和 QQ 机器人、识图模型、图库 Web 继续在 tx 运行。devbox 常驻奶龙、奶蛙两个固定 GPT-SoVITS 进程，分别返回 WAV；tx 在句首加播放缓冲，飞书使用 Opus 回复，QQ 使用 WAV 语音回复。

```text
飞书 / QQ 用户 → tx 原机器人 → 127.0.0.1:19880 → 奶龙 9880
                             └→ 127.0.0.1:19881 → 奶蛙 9881
                                      反向 SSH 隧道
```

开发机主动连接 tx，断线后由 systemd 重连。SSH 专用账号只允许监听 tx 的 `127.0.0.1:19880` 和 `19881`，不能执行命令或建立其他转发。HTTP 接口另用随机 Bearer 令牌校验请求。

## 使用

群聊中发送 `@机器人 奶龙说 今天的鸡腿都是我的！`；与机器人私聊时发送 `奶龙说 今天的鸡腿都是我的！`。

飞书和 QQ 都支持 `奶龙说`、`奶蛙说` 和 `奶蛙大笑`。
奶蛙音色使用已授权的八级哥笑饿魔参考音和 GPT-SoVITS v2ProPlus 官方基础权重，
奶龙、奶蛙权重分别固定在独立进程中，不再按请求热切换。
只支持中文台词，可包含数字和标点，最多 200 字。输入英文会收到中文限制提示。
QQ 沿用现有被动回复规则，语音关联触发消息，保留 5 分钟回复窗口和消息序号；
没有额外开启主动发言。

语音请求按以下限制执行，超限会直接返回提示，客户端不会自动重试。

| 位置 | 限制 |
| --- | --- |
| tx 机器人入口 | 每人间隔 30 秒，同一群或私聊间隔 5 秒，最多 4 个请求等待后端结果 |
| devbox 后端 | 只执行 1 次推理，最多 3 个等待位，按提交顺序处理 |
| tx 语音调度 | 每个音色最多等待 3 个任务，单消费者；15 秒未完成则只保留文字 |
| devbox 后端 | 每个音色仅执行当前任务，不再额外排队；每 60 秒最多接纳 12 个请求 |
| devbox systemd | 每个模型 CPU 配额 8 核，内存软限 8 GiB、硬限 10 GiB |

tx 为奶龙、奶蛙分别维护一个单消费者队列。连续失败三次后熔断，恢复等待按 60、120、240、300 秒递增；半开状态只放行一个探测任务。队满、熔断或 15 秒配对超时时，业务文字仍立即发送，过期语音不会稍后补发。

用户冷却按平台提供的身份计算；QQ 的群成员身份按群隔离，跨群不会强行合并为同一个人。所有平台的请求仍受同一后端全局频率和排队上限约束。

这里沿用已试听的社区 GPT-SoVITS v2ProPlus 权重，配合固定参考录音；没有进行新一轮训练。权重版本、上游提交和文件校验值记录在发布包的 `release.json`、`SHA256SUMS` 中。样本自动转写尚未经过人工校对，不能当作已经完成的训练集。

## 安装和更新

项目源码以当前工作区为准，服务器只保存运行副本。首次打包需要本地已有 `outputs/nailong-voice-20261102/gpt-sovits` 中的权重和上游源码。

```bash
python3 scripts/build_tts_release.py --output artifacts/tts-release
bash scripts/deploy_tts.sh devbox artifacts/tts-release
bash scripts/connect_tts_tunnel.sh devbox tx
bash scripts/deploy_voice_client.sh tx
```

打包命令拒绝覆盖已有目录。更新时请指定新的输出目录，并将它传给部署脚本。

开发机访问 PyPI 较慢时，可以预先下载 Python 3.11、Linux x86_64 依赖，传入开发机安装脚本的 `TTS_WHEELHOUSE` 环境变量；安装脚本支持离线安装。Torch 和 torchaudio 固定为 2.8.0 CPU 版，其他直接依赖见 `requirements.txt`。实际安装版本另外保存为 `/srv/nailong-tts/requirements.installed.txt`。

tx 只额外安装 PyAV 16.1.0，使用其内置 Opus 编码器。客户端源码按内容哈希放入 `/opt/nailong-bot/voice-releases/`，通过机器人服务的 systemd drop-in 选择新入口。原来的源码、虚拟环境中的其他依赖、模型、数据库和图库进程继续保留。客户端安装脚本只重启 `nailong-bot`。

tx 在 Opus 编码前加 350 毫秒的句首静音缓冲，保留原始音节；上传飞书时显式携带总时长，单位为毫秒。SDK 的普通音频上传路径不会传这个字段，因此带 `duration_ms` 的语音由飞书 adapter 单独上传，再复用 SDK 发送文件引用。旧消息的时长不会随代码更新而自动修正。

QQ 复用同次合成保留的 PCM WAV，包含相同的 350 毫秒缓冲，直接按 `file_type=3` 上传并通过 `msg_type=7` 回复。使用 base64 上传，`srv_send_msg=false`，不需要公开音频 URL，也不会为 QQ 重复合成。语音只记录为音频资源，不加入图片引用缓存。[腾讯官方 SDK 媒体流程](https://github.com/tencent-connect/qqbot-nodejs/blob/main/USAGE.md#10-媒体上传与发送)可供对照。

## 运行文件

| 机器 | 路径或服务 | 用途 |
| --- | --- | --- |
| devbox | `/srv/nailong-tts/releases/`、`current` | 发布包和当前版本链接 |
| devbox | `/srv/nailong-tts/.venv` | 独立 Python 环境 |
| devbox | `/etc/nailong-tts.env` | 服务端令牌，root 和服务组可读 |
| devbox | `/var/lib/nailong-tts-*`、`/var/cache/nailong-tts-*` | 两个模型的独立状态和缓存 |
| devbox | `nailong-tts@nailong.service` | 固定奶龙模型，本机 9880，内存上限 10 GiB |
| devbox | `nailong-tts@naiwa.service` | 固定奶蛙模型，本机 9881，内存上限 10 GiB |
| devbox | `nailong-tts-tunnel.service` | 连接 tx 的反向隧道 |
| tx | `/etc/nailong-tts-client.env` | 后端地址与客户端令牌，仅 root 可读 |
| tx | `/etc/systemd/system/nailong-bot.service.d/90-voice-client.conf` | 为原机器人选择语音分支入口和配置 |
| tx | `/etc/ssh/sshd_config.d/90-nailong-tts-tunnel.conf` | 限制隧道账号的转发和会话权限 |

不要把环境文件或私钥放进 Git，也不要在诊断输出中打印它们。原 `/etc/nailong-bot.env` 不需要改动。

## 检查和回滚

```bash
ssh devbox 'systemctl status nailong-tts@nailong nailong-tts@naiwa nailong-tts-tunnel --no-pager'
ssh tx 'systemctl status nailong-bot nailong-web --no-pager'
ssh tx 'curl --fail http://127.0.0.1:19880/healthz'
ssh tx 'curl --fail http://127.0.0.1:19881/healthz'
ssh devbox 'sudo -n journalctl -u "nailong-tts@*" -n 50 --no-pager'
```

`/healthz` 只在模型加载并真实预热合成成功后才可访问，提供 `busy`、`queued`、`queue_capacity` 和全局频率上限。`POST /tts` 需要 Bearer 令牌，JSON 为 `{"text":"你好"}`，可附整数 `seed`。响应为单声道、32 kHz、16 位 PCM WAV，响应头包含排队与生成耗时；未授权返回 401，参数错误返回 400，队满、排队超时或全局限流返回 429，并附 `Retry-After`。接口不提供模型上传或任意文件访问。

后端实例使用 `TTS_VOICE` 固定音色；`TTS_MAX_WAITING=0`，排队由 tx 调度层负责。客户端读取超时为 120 秒，覆盖合成；连接超时为 5 秒。

回退到原机器人入口：

```bash
ssh tx 'rm /etc/systemd/system/nailong-bot.service.d/90-voice-client.conf; systemctl daemon-reload; systemctl restart nailong-bot'
```

此操作保留客户端代码副本和配置，图库服务无需重启。如果只想暂停语音后端，可停止开发机的两个 `nailong-tts@*` 实例和 `nailong-tts-tunnel`；机器人会自动只保留文字回复。回退后端版本时，将 `/srv/nailong-tts/current` 切到已验证的旧发布目录，再重启两个实例。
