# 奶龙奶蛙识别机器人

这是一个识别奶龙、奶蛙和其他图片的机器人项目。保留 MobileNetV3-Small 基线；生产候选使用微调 DINOv2 与联合分类头，优先减少奶蛙漏识别。推理只执行神经网络前向和概率计算，不做图库检索。

2026-10-31 的奶蛙优先版本通过 `MILK_MODEL_PATH` 加载可移植的 `manifest.json`，主分类和豆包过滤共享模型。文档奶蛙识别为 61/66，压缩图为 58/66，群聊奶蛙为 19/21；普通图误收有所增加。这些是已使用过的诊断回归集，不能作为独立的 95% 准确率证明。

## 数据目录

```text
data/
├── train/
│   ├── doubao/
│   ├── nailong/
│   ├── naiwa/
│   └── other/
└── test/
    ├── doubao/
    ├── nailong/
    ├── naiwa/
    └── other/
```

原 `train/test` 目录保留 279 个训练样本和 53 张测试图。其中豆包 43 张（训练 33、测试 10），包含原版头像、保留原形象的表情图、风格二创，以及奶龙/奶蛙化豆包；2026-10-26 新增下载 40 张，另将原 `other` 中 3 张豆包重新标注。逐图来源、子类型及内容指纹见 `data/DOUBAO_20261026.csv`。

2026-10-30 从线上下载 823 张图片，复核去重后新增 305 张训练资料，结合旧资料建立独立的三分类数据目录 `data/cnn-20261030/`，共 636 张：训练 518 张，验证与测试各 59 张。主分类器训练时把原豆包资料归入 `other`，独立豆包检测器保持不变。逐图来源与划分见 `data/RETRAIN_20261030.csv`，新旧模型对比与运行门槛见 `artifacts/cnn-20261030/report.md`。

本轮训练 33 轮，第 17 轮为最佳验证模型，已部署。独立测试集准确率从 79.7% 提升到 83.1%；运行门槛从 65% 降到 50% 后，保留豆包拦截时正确接纳奶图从 9/26 增加到 16/26，普通图片误收从 0/33 增加到 2/33。存量刷新重新归类 112 张，其中 102 张从“其他”进入奶龙/奶蛙分类；人工纠正保留。

训练时会对 GIF 随机抽帧，并提高动画样本的采样权重；线上新增图在进入训练集和测试集前会人工复核，并用内容指纹排除近重复泄漏。本轮奶龙训练图只有 37 张，验证与测试各只有 3 张，相关指标波动较大，后续仍需补充真实奶龙的复杂背景、截图压缩和局部遮挡样本。

## 本地开发

项目使用 Python 3.11 和 `uv`：

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

训练奶龙/奶蛙/其他分类器（默认保留原结构，并加强增强与正则）：

```bash
uv run nailong-train
```

2026-10-30 首轮增量训练使用的主要配置如下（当前训练器还修正了冻结层的 BatchNorm 统计更新）：

```bash
uv run nailong-train \
  --data-dir data/cnn-20261030 \
  --validation-dir data/cnn-20261030/val \
  --initial-model artifacts/cnn-20261030/baseline-model.pt \
  --output-dir artifacts/cnn-20261030/candidate \
  --architecture mobilenet_v3_small --dropout 0.2 \
  --augmentation-profile color_invariant_v2 --weight-decay 5e-4 \
  --selection-metric accuracy \
  --epochs 64 --patience 16 --batch-size 16 --num-workers 2 \
  --class-balance-power 0.85 --label-smoothing 0.10 \
  --learning-rate 1e-4 --feature-learning-rate 3e-5 \
  --threshold 0.50 --seed 20261030
```

该目录的验证与测试图片都来自新收集的线上图片，不参与训练。旧 `test` 图片曾参与历史训练，本轮按训练资料使用，不能拿旧测试集成绩证明泛化效果。

针对过拟合的训练入口：

```bash
uv run nailong-train \
  --data-dir data/cnn-20261030 \
  --validation-dir data/cnn-20261030/val \
  --initial-model artifacts/model.pt \
  --output-dir artifacts/cnn-regularized-20261030/conv-pool \
  --architecture mobilenet_v3_small_regularized \
  --augmentation-profile chat_robust_v3 \
  --epochs 40 --patience 10 --batch-size 16 \
  --learning-rate 5e-4 --feature-learning-rate 3e-5 \
  --weight-decay 5e-3 --dropout 0.4 --label-smoothing 0.1 \
  --max-grad-norm 1.0 --unfreeze-blocks 2 \
  --class-balance-power 0.85 --selection-metric macro_f1 \
  --threshold 0.5 --seed 20261030 --skip-test-evaluation
```

增强只作用于训练视图：保留裁剪、旋转、颜色扰动和遮挡，新增缩小再放大、JPEG 重压缩与轻度模糊，不增加原始样本数。AdamW 对卷积与线性权重施加衰减，偏置和归一化参数不惩罚；训练还使用梯度裁剪，并冻结不参与微调的 BatchNorm 均值与方差。

新结构复用 MobileNetV3-Small 骨干，新增深度卷积、1×1 通道压缩、GroupNorm 和平均池化，把原来的 576→1024 分类头缩为 128→128。主分类器训练的 Dropout 默认为 0.4。旧 checkpoint 仍可加载，独立豆包训练器保留原结构和增强默认值。切换结构继续训练时仅迁移骨干权重，新分类头重新学习。

对照组使用同一命令，把 `--architecture` 改为 `mobilenet_v3_small`、输出目录改为 `artifacts/cnn-regularized-20261030/control`。两个候选先按验证集宏平均 F1 选择，再评估已固定的模型：

```bash
uv run python scripts/evaluate_generalization.py \
  --experiment-dir artifacts/cnn-regularized-20261030 \
  --data-dir data/cnn-20261030 \
  --baseline artifacts/model.pt \
  --doubao-model artifacts/doubao-detector-20261026/model.pt
```

选择在 `selection.json` 中先行记录，测试结果写入 `comparison.json`。评估同时使用原图和固定压缩/缩放/模糊视图；后者复用同一批图片，只能说明对这些退化的耐受程度。训练和评估均不自动替换运行模型。

本轮两组训练已完成。按验证集选择的是原结构加强正则的候选：原图测试准确率 83.1%→89.8%，保留豆包拦截时正确接纳奶图 16/26→20/26，普通图误收仍为 2/33。压缩、缩放和模糊视图上正确接纳奶图 15/26→19/26，但普通图误收 2/33→3/33，宏平均 F1 略降。新增卷积、池化的模型验证宏平均 F1 为 0.873，低于原结构加强正则的 0.916，因此新结构保留为显式可选方案，没有设为默认或替换线上权重。

结构、训练配置、完整结果及限制见 `artifacts/cnn-regularized-20261030/report.md`，增强预览与训练曲线保存在同目录。这份测试划分已在上一轮使用，奶龙测试图只有 3 张，不能据此宣称全新群聊资料上的泛化问题已解决。

训练独立豆包检测器：

```bash
uv run nailong-doubao-train --output-dir artifacts/doubao-detector-20261026
```

豆包模型将 `doubao` 作为正例，将奶龙、奶蛙和其他图片作为负例，输出“含豆包 / 未检测到豆包”。它的权重与原三分类模型互不兼容，保存在独立目录；接入机器人时应在原分类前先执行豆包检测。

当前训练命令默认以豆包概率 `0.30` 作为拦截阈值。该阈值优先降低漏检风险，但在接入群聊前仍需用更多含豆包局部贴纸、同框和复杂截图样本验证。

2026-10-21 的线上增量模型从上一版继续低学习率微调，下面是当时使用三分类数据的历史命令：

```bash
uv run nailong-train \
  --initial-model artifacts/model.20261021-pre-online-expansion.pt \
  --architecture mobilenet_v3_small --dropout 0.2 \
  --augmentation-profile color_invariant_v2 --weight-decay 5e-4 \
  --selection-metric accuracy \
  --class-balance-power 0.75 \
  --learning-rate 5e-5 \
  --feature-learning-rate 5e-6
```

豆包检测器的产物保存在：

```text
artifacts/doubao-detector-20261026/model.pt
artifacts/doubao-detector-20261026/metrics.json
```

测试单张图片：

```bash
uv run nailong-predict data/test/nailong/image.png
```

## 机器人接入层

接入结构为 `飞书 / QQ → adapters → 统一消息与事件 → 共用业务服务`。`bot.py`
负责识图、接龙、纠正、猜图、日报和新人起奶名；SDK、平台 ID 转换、引用消息读取、
图片收发和通讯录查询放在 `src/nailong_classifier/adapters/`。所有接入实例共用
`BackendServices` 中的一份模型和 `BotStore`，当前通过进程内调用共用后端，尚未拆成
独立 HTTP 服务。多个飞书 SDK 运行在各自的子进程中，避免其全局长连接事件循环互相干扰；
模型推理与业务状态留在主进程。

不配置 `BOT_ADAPTERS_CONFIG` 时，原来的 `LARK_APP_ID` / `LARK_APP_SECRET` 启动方式
继续生效。接入多个机器人时，复制 `deploy/bots.example.toml`，按需保留其中的实例，
然后启动：

```bash
uv run nailong-bot --bots-config /etc/nailong-bots.toml
# systemd 部署也可在 /etc/nailong-bot.env 中设置：
# BOT_ADAPTERS_CONFIG=/etc/nailong-bots.toml
```

每个飞书实例配置自己的 App ID / Secret 环境变量名；QQ 使用腾讯官方机器人平台的
AppID / AppSecret，通过官方 WebSocket 接收事件、API v2 回复消息。示例已配置 QQ
机器人“我是奶龙”（AppID `102498836`），密钥由 `QQ_APP_SECRET` 环境变量提供。
后台入口：[QQ 机器人管理](https://q.qq.com/qqbot/dashboard/manage/102498836?sec=account-info)。
在开发设置中选择 WebSocket，并配置测试群/好友或完成发布。连接时会自动申请和刷新
AccessToken、维持心跳、断线恢复，不需要另外运行 QQ 登录网关。

QQ 支持群 @和 C2C 消息、图片识别和回复；奶运、排行榜使用文字展示，图鉴合成一张图。
群里直接发图片并 @机器人，会接一张同类奶图，同时展示识别、奶含量和稀有度；
回复图片并 @机器人也会接龙并展示识图结果，私聊保持识图回复。
QQ 奶家人玩法使用 `@机器人 接奶`（也支持 `奶家人`）：按发消息的人本人登记，
并回复一张随机奶图。私聊直接发 `接奶` 即可，不需要 @群友、输入昵称或提前发言；
图鉴为空时提示先发奶图收录。飞书保留原来的奶家人点名玩法。
飞书和 QQ 都支持 `@机器人 求赐名`，私聊直接发 `求赐名`：随机组合定语、洋气
名字和奶龙/奶蛙称号，例如“尼古拉斯·奶龙影子里的第二张嘴”
“奶龙肚皮里传来的敲门声·韩奶清”“奶蛙笑声里多出来的那个人”。
词库混合荒唐职位和猎奇场景，还能抽到“奶龙鸡腿失踪案嫌疑人”
“奶蛙替补席永久居民”，或“住在奶龙第三个胃里的”这样的定语。
每类词库候选都明确带“奶龙”或“奶蛙”。
词库位于 `src/nailong_classifier/milk_name_words.py`，按 2026-10-30 的抖音
[奶龙](https://www.douyin.com/search/奶龙)、
[奶龙 小七](https://www.douyin.com/search/奶龙%20小七)、
[奶蛙](https://www.douyin.com/search/奶蛙)、
[奶蛙 捧腹大笑](https://www.douyin.com/search/奶蛙%20捧腹大笑)搜索文案扩充。
大侠、大胃王、鸡腿、暴暴龙、小七、避雷师、大笑、上掌下拳、通透等是检索到的梗；
组合称号包括沿这些主题原创的笑话，不代表抖音原句、官方设定或热度排名。
词库以角色本身为中心：奶龙写贪吃、鸡腿、小七、暴暴龙、天线和卷风，
奶蛙写大笑、扣篮、上掌下拳和通透。另参考
[我是奶龙](https://www.douyin.com/search/奶龙%20我是奶龙)搜索中的真假奶龙文案，
改成“我才是奶龙的鸡腿证人”“奶龙冒充奶蛙被饭量出卖”等身份闹剧。
曼波、哈基米、咕咕嘎嘎、蓝白碗等与角色无直接关系的通用梗已从候选词库移除；
同场出现在搜索结果里，不作为属于角色的依据。
现在有 118 个称号、52 个定语和 20 个洋气名字，保留原来的猎奇场景，
补入“奶龙减肥第一天的第八顿”“奶龙把小七冰箱吃出回音”
“奶蛙捧腹大笑捧丢了肚子”“奶蛙通透得只剩笑声”等角色笑话。
起名读取本地词库，无需临时联网搜索。
词库更新不会改掉已保存的奶名，再次求赐名才会抽取新名字。
再求一次就换名，连续两次不会相同；同库所有用户的当前奶名唯一，撞名会重抽，
并由数据库唯一索引兜底。词库候选被占满时会追加随机短标记；旧版重复奶名在
升级时保留先赐的一份，给后赐的名字补随机短标记，原昵称和用户对应关系保留。
使用昵称的一类名字会随机调整“奶”字位置，例如韩婉清可以变成奶婉清、韩奶清、
韩婉奶，也可以前后加“奶”；名字本来有“奶”字就保留主体。新人入群也使用同一
套随机定语、洋气名字和奶系称号，不再固定替换最后一个字。保存首次原昵称与
最新奶名的对应关系，改昵称或服务重启不会自动换名。入群欢迎共用这份映射，
已有奶名会保留，事件重投或欢迎失败后重试也不会重新起名，新人重名时同样查重。拿不到昵称
时先尝试查询，仍拿不到则以“神秘奶友”起名，不要求手动输入昵称。
飞书昵称查询依次使用消息显示名、已记住的 @ 昵称、通讯录和群成员列表；
通讯录无权限时仍可用群成员接口兜底。成功的群成员查询缓存五分钟，失败缓存
一分钟，避免反复请求；查询超时或都无权限时仍正常生成随机奶名。后续消息、
@ 或群成员信息提供真实昵称时，只补全原昵称对应关系，不自动更换奶名。
QQ 继续使用消息携带的昵称，没有群成员或独立昵称查询接口。
普通群通常只投递 @消息，猜图答案需发送 `@机器人 奶龙/奶蛙`。群主授权全量消息后，
普通图片也能自动接龙、普通文字也可参与抢答。纠正分类通过回复原图或接龙图发送
`@机器人 纠正为奶龙/奶蛙/其他` 完成。引用兼容旧 `message_reference` 与新版
`message_scene` 引用索引；原消息不在缓存时，若 QQ 在 `msg_elements` 中附带引用图片，
仍可识别；未附图片的未知历史引用需重新发送。飞书保留卡片及按钮。

QQ 官方群接口不提供普通成员名单、昵称查询和新人入群通知，因此新人起奶名仍只在飞书启用；
QQ 用户发过消息后会登记为奶家人，使用事件携带的昵称，缺少昵称时显示“神秘奶友”。默认关闭群主动推送，
仅在五分钟内利用收到的消息进行回复；定时日报等需要群主授权并在该实例设置
`allow_proactive = true`。C2C 不擅自发送无触发消息。真实 QQ 和多飞书账号仍需配置凭证后
进行聊天验收。

新增实例的群、用户和消息 ID 按平台及实例名隔离；图库素材仍按图片内容共享。日报、猜图、
欢迎去重和接龙轮换按各自的会话统计。实例 `id` 需保持稳定，改名会产生新的身份命名空间。
`legacy = true` 只允许给原来的一个飞书实例配置，用于沿用历史数据；QQ 和新增飞书实例
不能使用它。

多实例模式下，图鉴文档配置写在所属飞书实例的 `catalog_document_id` /
`catalog_section_block_id` 中；同一文档只允许一个同步实例。QQ 收录的新图也会通知该实例同步。
单实例旧配置仍读取 `CATALOG_DOCUMENT_ID` / `CATALOG_SECTION_BLOCK_ID`。

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
- 自动接龙的奶图也有 20% 概率转成闪亮 GIF，这个概率直接作用于每次接龙，不受主动推送的两小时冷却影响。GIF 在飞书聊天内循环播放；原图接龙保留原有说明。
- 需要查看文字识别结果时，回复原图片并 `@` 机器人。`@` 识别时除了识别结论，还会附上「奶龙浓度检测仪」读数：模型奶含量（奶龙和奶蛙概率之和）占 60%，可见黄色像素占比占 40%。黄色检测排除白灰色、暗色和透明区域，使用完整图片当前帧。奶含量是娱乐读数，分类置信度仍按原模型判断；奶龙/奶蛙图再附一行「稀有度」评级。
- “其他”图的加权奶含量达到 40% 时，随机回复“这不是奶娃吧？！”等害怕漏识别的文案，模型不确定时仍提示不确定；人工纠正的分类优先。
- 稀有度按这张图在群里被发送的累计次数评定：发得越少越稀有，只冒过一两次是传说，天天刷屏就是烂大街。
- 取 CNN 概率最高的类别；奶龙或奶蛙达到 50% 时接龙并收进图鉴，低于 50% 时群聊静默。运行环境中的 `MODEL_THRESHOLD` 可覆盖模型保存的门槛。
- 单聊中遇到“其他”或低置信结果时也会返回文字提示；群聊保持静默，避免刷屏。
- 没见过的图片按 SHA-256 去重后收入图库，同一张图重复发送不会重复收录。
- 回复原图或机器人的接龙图，发送 `纠正为奶龙`、`纠正为奶蛙` 或 `纠正为其他`，可以修正分类。
- 每天北京时间 23:55，机器人会向当天出现过图片或表情的群自动发送“今日含奶量”。发送 `今日含奶量` 也可以随时手动查看。
- 发送 `奶龙图鉴`，查看群聊最近收集的 9 张奶龙或奶蛙图片。只有模型通过置信度门槛，或经人工明确纠正为奶龙/奶蛙的图片才进入图鉴；“其他”和不确定图片不进入图鉴。图鉴不计算接龙种子，每张图片都可以单独查看。
- 配置图鉴文档后，新图片首次进入图鉴时会自动追加到目标章节的最后一个四列网格。重复图片、最高候选为“其他”的图片不会追加；同步失败会保留进度并自动重试。
- 发送 `猜图`（或 `盲盒猜图`）开一局盲盒猜图：机器人从图鉴里随机抽一张奶图，裁出局部并打上高斯模糊出题，群友直接发 `奶龙` 或 `奶蛙` 抢答，本群第一个答对的人得 1 分并亮出完整原图。发 `公布答案` 揭晓本局但不给任何人加分，发 `猜图榜` 查看本群积分排行。积分按群聊独立统计。
- 发送 `今日奶运`（也支持 `奶运`、`奶龙签`、`抽奶签`）抽当天奶签，获得奶气值、接梗力、宜忌、关键词和群聊称号。同一人在同一群同一天结果固定，第二天自动换签。
- 发送 `@我是奶龙 help` 或 `@我是奶龙 帮助`，查看完整玩法。
- 只要在机器人所在的群里发过消息，就会被登记为奶家人（只保存飞书用户 ID 的哈希和可用昵称）。同一条消息里 `@我是奶龙` 再 `@一位群友`，机器人会判断这位群友是不是奶家人；命中后回复一张奶图，没命中时提示先在群里冒个泡。
- `@我是奶龙 #奶蛙`、`@我是奶龙 #文艺复兴` 或 `@我是奶龙 #豆奶` 可以立即按指定话题搜索并回复一条抖音视频；一次最多识别三个话题。
- 配置 `DOUYIN_COOKIE_FILE` 后，机器人搜索指定抖音话题，只有结构化标签中明确含“奶龙”或“奶蛙”、且封面识别通过的视频才会缓存 15 分钟并发送；Cookie 只保存在服务器受限路径，不进入仓库或日志。群聊普通消息有 40% 概率触发视频链接，同一群按滚动一小时最多发送 2 条；链接发送成功时不会再触发本条消息的随机奶图。
- 群聊普通消息有 3% 概率触发主动奶图，同一群冷却 2 小时，始终只发图片。其中默认 20% 的奶图转成短循环闪亮 GIF，在聊天里模拟高光闪一下；转换或上传失败发原图，原图库文件不覆盖。
- 发送 `HDR奶图`、`HDR奶龙` 或 `来张HDR奶图`，立即从奶龙/奶蛙图池选图并回复闪亮 GIF，不用等随机触发。GIF 上传失败时回退原图，发送日志记录 `flash_gif=True/False`。

普通文字可触发上述随机发图，不会收到文字回复。每条消息只处理第一张图，超过 20 MB 的图片不会进入识别。

闪亮 GIF 的触发概率仍通过 `RANDOM_PUSH_HDR_PROBABILITY`（0~1，设 0 关闭接龙和主动推送的随机转换，手动指令仍可使用）调整。`HDR_PEAK_NITS` 只保留给本地 Ultra HDR 文件编码工具，不再参与飞书聊天发送。可用以下命令单独制作并保存真正的 HDR 图片：

视频链接通过 `DOUYIN_COOKIE_FILE` 配置的浏览器导出 Cookie 检索；`DOUYIN_SEARCH_KEYWORDS` 支持逗号分隔的话题（如 `#奶蛙,#文艺复兴,#豆奶`），机器人会在有结果的话题间等概率选择。每条候选必须带有“奶龙”或“奶蛙”结构化标签，并通过封面分类校验；标题文本或未校验的静态链接不能作为推送来源。`DOUYIN_SEARCH_CACHE_SECONDS` 默认 `900`。`DOUYIN_VIDEO_PUSH_PROBABILITY` 默认 `0.4`，`DOUYIN_VIDEO_PUSH_WINDOW_SECONDS` 默认 `3600`，`DOUYIN_VIDEO_PUSH_MAX_PER_WINDOW` 默认 `2`。次数按群聊持久化记录，服务重启后仍会继续限流。

```bash
uv run nailong-hdr path/to/milk.png /tmp/milk-hdr.jpg --peak-nits 1000
```

## Web 图鉴

Web 图鉴只公开奶龙和奶蛙图片，支持分页和按收录时间、出现次数排序。“其他”分类只保留在后台，用于模型判断和人工纠错，不会出现在 Web 列表、统计或图片接口中：

```bash
uv run nailong-web --state-dir runtime
```

本机打开 `http://127.0.0.1:8765`。列表页只请求最长边 480px 的 WebP 缩略图，每页 36 张；缩略图首次生成后保存在 `runtime/preview-cache/thumbnails`，并带一年期 immutable 浏览器缓存。只有点开单图详情时才读取原图，原图同样支持 ETag 和长期缓存。

线上 `nailong-web` 服务只监听 `tx` 的 `127.0.0.1:8765`。开发机 `devbox` 运行常驻后端代理，对内提供 `http://10.36.7.51:8765/`；公开缩略图和静态资源缓存在 `/var/cache/nailong-gallery-proxy`。`tx` 的公网 `/nailong/` 只接受开发机代理携带的专用令牌，直接访问返回 404。

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

奶蛙优先版本需要 `artifacts/milk-release-20261031/` 下的清单和六份权重。
本地实验完成后运行 `uv run python scripts/build_milk_release.py` 生成部署包；
从 GitHub 恢复时，下载备份 Release 中的 `milk-release-20261031.tar.gz`，
在项目根目录解压即可。运行时不需要实验脚本、训练数据或 Hugging Face 在线下载。

清单固定分类门槛 `0.50` 和豆包过滤门槛 `0.30`；旧环境覆盖值不同会导致启动报错，
以免偏离验收版本。清单中每份权重都带 SHA-256，启动日志记录实际加载版本。

GitHub 备份保存源代码和模型包。数据库、图库原图、环境密钥与 Cookie 保存在服务器
`/var/backups/nailong-bot/` 下的受限完整备份中，不上传公开仓库。
回滚代码、权重、依赖和配置时，在服务器以 root 运行
`bash /opt/nailong-bot/deploy/rollback.sh /var/backups/nailong-bot/full-TIMESTAMP`。
该命令保留上线后的聊天记录与图库；如需恢复数据库，必须另行停服务恢复。

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
systemctl enable --now nailong-web
systemctl status nailong-bot
systemctl status nailong-web
journalctl -u nailong-bot -f
```

凭据文件位于 `/etc`，不会同步回本地，也不会进入 Git。

## Git 约定

本地目录是唯一源码工作区。服务器只保存运行副本，不直接修改代码。训练图片、模型权重、虚拟环境和凭据均已加入 `.gitignore`。
