# 机器人应用分层

## 调用方向

```text
FeishuAdapter / QQAdapter
        ↓
MessageContext（含 event_type）
        ↓
FeatureRouter → FeatureHandler
        ↓
ResponsePlan / AsyncTask
        ↓
WorkScheduler / Domain Service
        ↓
DeliveryGateway
        ↓
FeishuAdapter / QQAdapter
```

Adapter 只处理平台协议、身份隔离、资源上传下载和消息收发。Adapter 不选择具体业务功能。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `application/context.py` | 与平台无关的消息、资源和提及上下文 |
| `application/router.py` | 按显式优先级选择一个主功能或事件处理器 |
| `application/features/` | 命令匹配与业务用例编排 |
| `application/response.py` | 描述待发送消息与异步任务 |
| `application/work_scheduler.py` | 有界队列、超时、熔断和降级 |
| `application/inference_dispatch.py` | 奶图和豆包模型的全局串行推理 |
| `application/dispatch.py` | 消息发送并发、每群顺序和语音配对 |
| `domain/` | 不依赖平台与网络的业务规则 |
| `integrations/` | TTS、社交平台、图鉴文档等外部能力 |
| `adapters/` | 飞书和 QQ 的协议实现 |

## 强制边界

- `application/features/` 禁止导入 `bot`、`adapters`、`persistence` 和具体 `integrations`。
- Feature 禁止直接调用 `send`、`update_card`。
- Feature 禁止自行调用 `asyncio.create_task`、`asyncio.to_thread` 或创建锁。
- 高消耗工作只能作为 `AsyncTask` 提交给 `WorkScheduler`。
- Feature 只能返回 `ResponsePlan`；发送由 `DeliveryGateway` 执行。
- 一条消息或业务事件只选择一个主 Feature；主动推送属于独立后台任务。
- 新功能必须新增或扩展 Feature，不得继续向 `handle_command` 添加分支。

边界由 `scripts/check_architecture.py` 扫描，违规会输出文件和行号。

## 迁移顺序

| 状态 | 功能 |
| --- | --- |
| 已迁移 | 手动初始化、帮助、安静/恢复、语音点播、社交链接导入 |
| 已接入统一基础设施 | 全部聊天发送、卡片更新、奶图分类、豆包检测、主动推送 |
| EventRouter 过渡 | 新人欢迎、卡片纠正 |
| Legacy 待迁移 | 奶名、奶签、奶字、企业表情、接龙、HDR、怪图、猜图、排行、图鉴、日报设置 |

迁移期间 `LegacyCommandFeature` 以最低优先级承接尚未迁移的命令。每迁移一项，必须同时删除
`handle_command` 中对应分支；当清单完成后删除 Legacy Feature 和旧总路由。
