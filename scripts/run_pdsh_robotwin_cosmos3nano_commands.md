# RoboTwin Cosmos3-Nano 启动命令汇总

## 可用脚本

### 1. 主脚本 - 完整训练
**脚本路径**: `scripts/run_pdsh_robotwin_cosmos3nano.sh`
**用途**: 完整的 RoboTwin/LeRobot 动作 SFT 训练

**默认配置**:
- `MAX_ITER`: 10000 步
- `SAVE_ITER`: 1000 步
- `OBSERVATION_IMAGE_MODE`: concat（单图拼接）
- 使用所有 episode 数据

**启动命令**:
```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

TRAINING_NAME=concat_view \
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 \
MASTER_ADDR=29.191.210.123 \
MASTER_PORT=50121 \
NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
ROBOTWIN_MODE=policy \
OBSERVATION_IMAGE_MODE=concat \
ROBOTWIN_VIEWPOINT=concat_view \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 \
WANDB_GROUP=robotwin_lerobot_action_sft \
WANDB_NAME=cosmos3nano_robotwin_concat_view_8n64g_${RUN_STAMP} \
MAX_ITER=10000 \
SAVE_ITER=1000 \
bash scripts/run_pdsh_robotwin_cosmos3nano.sh
```

### 2. One-Task 脚本 - 单 Task 过滤训练
**脚本路径**: `scripts/run_pdsh_robotwin_cosmos3nano_one.sh`
**用途**: 只训练一个 RoboTwin task 下的所有 episodes，并重复采样用于 overfit

**默认配置**:
- `MAX_ITER`: 1000 步
- `SAVE_ITER`: 200 步
- `ROBOTWIN_TASK_INDEX`: 0（默认过滤第 0 个 task）
- `ROBOTWIN_DATASET_REPEAT`: 1000（重复该 task 数据，避免小数据集读空）
- `ROBOTWIN_VIEWPOINT`: concat_view（三相机基线）
- `ROBOTWIN_USE_STATE`: true（把 14D `observation.state` qpos/proprio 作为 action 序列第 0 行条件）
- `ROBOTWIN_STATE_KEY`: observation.state
- `OPTIMIZER_LR`: 2.0e-4
- `SCHEDULER_F_MAX`: 1.0
- `SCHEDULER_F_MIN`: 0.0（从 2.0e-4 继续衰减）

**启动命令**:
```bash
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

TRAINING_NAME=one_task_concat_view \
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 \
MASTER_ADDR=29.191.210.123 \
MASTER_PORT=50121 \
NPROC_PER_NODE=8 \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
ROBOTWIN_MODE=policy \
OBSERVATION_IMAGE_MODE=concat \
ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_TASK_INDEX=0 \
ROBOTWIN_DATASET_REPEAT=1000 \
ROBOTWIN_USE_STATE=true \
ROBOTWIN_STATE_KEY=observation.state \
OPTIMIZER_LR=2.0e-4 \
SCHEDULER_F_MAX=1.0 \
SCHEDULER_F_MIN=0.0 \
SCHEDULER_WARM_UP_STEPS=0 \
COSMOS_MODELS_ROOT=/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models \
WANDB_PROJECT=cosmos3 \
WANDB_GROUP=robotwin_lerobot_action_sft \
WANDB_NAME=cosmos3nano_robotwin_one_task_0_concat_view_8n64g_${RUN_STAMP} \
MAX_ITER=1000 \
SAVE_ITER=200 \
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh
```

## 关键参数说明

### 训练模式
- `ROBOTWIN_MODE=policy`: 策略训练模式

### 图像观测模式
- `OBSERVATION_IMAGE_MODE=concat`: 单图拼接模式（传统）
- `OBSERVATION_IMAGE_MODE=multi_image`: 多视角图像模式

### 数据过滤与重复
- `ROBOTWIN_TASK_INDEX=0`: 只使用 `task_index=0` 的所有 episodes
- `ROBOTWIN_TASK_NAME=xxx`: 按 `meta/tasks.jsonl` 里的 task 名称精确过滤
- `ROBOTWIN_DATASET_REPEAT=1000`: 将过滤后的数据长度重复 1000 倍，适合 one-task overfit
- `MAX_EPISODES=1`: 在过滤后最多保留 1 个 episode；one-task 训练一般不要设置
- `MAX_EPISODES` 未设置: 使用过滤条件下的所有可用 episodes

### Qpos / Proprio 条件
- `ROBOTWIN_USE_STATE=true`: 将首帧 14D `observation.state` 拼到 action 序列最前面，作为条件 action，不参与预测
- `ROBOTWIN_STATE_KEY=observation.state`: RoboTwin 数据中 qpos/proprio 字段；维度为 14，和 `action` 的 joint/gripper 顺序一致

### 学习率配置
- `OPTIMIZER_LR=2.0e-4`: optimizer base lr
- `SCHEDULER_F_MAX=1.0`: 初始倍率为 1.0，初始实际 lr 为 `2.0e-4`
- `SCHEDULER_F_MIN=0.0`: 保持 scheduler 衰减，默认从 `2.0e-4` 衰减到 `0`

### 训练配置
- `MAX_ITER`: 总训练步数
- `SAVE_ITER`: 保存 checkpoint 的间隔步数

### 网络配置
- `NNODES`: 节点数量
- `NPROC_PER_NODE`: 每个节点的 GPU 数量
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 主节点端口

## 常用组合

### 完整训练 - Concat View
```bash
TRAINING_NAME=full_concat_view \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
OBSERVATION_IMAGE_MODE=concat \
ROBOTWIN_VIEWPOINT=concat_view \
MAX_ITER=20000 \
SAVE_ITER=2000 \
bash scripts/run_pdsh_robotwin_cosmos3nano.sh
```

### 完整训练 - 单图拼接
```bash
TRAINING_NAME=full_concat \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
OBSERVATION_IMAGE_MODE=concat \
ROBOTWIN_VIEWPOINT=concat_view \
MAX_ITER=20000 \
SAVE_ITER=2000 \
bash scripts/run_pdsh_robotwin_cosmos3nano.sh
```

### 快速测试 - 单 Task Overfit
```bash
TRAINING_NAME=test_one_task \
COSMOS_WORKDIR=/mnt/lyn/workspace/wam \
OBSERVATION_IMAGE_MODE=concat \
ROBOTWIN_VIEWPOINT=concat_view \
ROBOTWIN_TASK_INDEX=0 \
ROBOTWIN_DATASET_REPEAT=1000 \
ROBOTWIN_USE_STATE=true \
ROBOTWIN_STATE_KEY=observation.state \
OPTIMIZER_LR=2.0e-4 \
SCHEDULER_F_MAX=1.0 \
SCHEDULER_F_MIN=0.0 \
SCHEDULER_WARM_UP_STEPS=0 \
MAX_ITER=500 \
SAVE_ITER=100 \
bash scripts/run_pdsh_robotwin_cosmos3nano_one.sh
```

## 节点配置示例

### 8 节点 64 GPU
```bash
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50 \
NNODES=8 \
MASTER_ADDR=29.191.210.123 \
MASTER_PORT=50121 \
NPROC_PER_NODE=8
```

### 4 节点 32 GPU
```bash
EXP_IP_LIST=29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187 \
NNODES=4 \
MASTER_ADDR=29.191.210.123 \
MASTER_PORT=50121 \
NPROC_PER_NODE=8
```

## 注意事项

1. **环境变量优先级**: 命令行设置 > 脚本默认值
2. **代码路径**: `COSMOS_WORKDIR` 默认使用 `/mnt/lyn/workspace/wam`，确保所有节点这个路径下是当前修改后的仓库
3. **数据路径**: 默认使用 `/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0`
4. **模型路径**: 默认使用 `/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/models`
5. **输出路径**: 默认输出到 `/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/outputs/train_robotwin_cosmos3nano/{TRAINING_NAME}`
6. **W&B 配置**: 默认项目为 `cosmos3`，分组为 `robotwin_lerobot_action_sft`

## 故障排查

- 确保所有节点可以访问共享存储
- 检查网络连通性和 NCCL 配置
- 验证环境变量设置正确
- 查看日志文件确认训练状态