# 双 8node 机器拼成 16node 启动说明

把两台 8-node 机器 (各 8 节点 × 8 GPU) 拼成一个 **16 node = 128 卡** 的训练 job。
全量 RoboTwin 数据训练,gbs=2048。

对应脚本:
- `run_all_task_lr2e-4_gbs2048_16node_A.sh` —— 在**机器 A (head, 含全局 rank0)** 上跑
- `run_all_task_lr2e-4_gbs2048_16node_B.sh` —— 在**机器 B** 上跑

---

## 一、怎么启动

**先 A 后 B**(让 master 先就位)。两台机器上都从含分片改动的 checkout 启动:

```bash
# 机器 A (head)
bash /mnt/lyn/wjh/workspace/cosmos3/wam/scripts/0623_exp/run_all_task_lr2e-4_gbs2048_16node_A.sh

# 机器 B (A 起来后再起)
bash /mnt/lyn/wjh/workspace/cosmos3/wam/scripts/0623_exp/run_all_task_lr2e-4_gbs2048_16node_B.sh
```

> 脚本内部会 `cd "$COSMOS_WORKDIR"` 再用绝对路径调 `run_pdsh`,所以**在哪个目录执行都可以**,不受当前 cwd 影响。

---

## 二、两台机器的参数对照

| 参数 | 机器 A | 机器 B | 说明 |
|---|---|---|---|
| `EXP_IP_LIST` | A 的 8 个 IP | B 的 8 个 IP | **各列自己的节点**,不是全部 16 个 |
| `NNODES` | `16` | `16` | 全局总节点数,两边一致 |
| `NODE_RANK_OFFSET` | `0` | `8` | 本机起始 global rank。A→rank 0-7,B→rank 8-15 |
| `MASTER_ADDR` | `29.119.84.104` | `29.119.84.104` | **必须同一个** = 机器 A 第一个 IP = 全局 rank0 |
| `MASTER_PORT` | `50221` | `50221` | 一致;rank0 节点上不可被占用 |
| `NPROC_PER_NODE` | `8` | `8` | 每节点 8 GPU |

**A 的 IP**:`29.119.84.104,29.127.48.51,29.127.50.141,29.232.241.61,29.119.96.20,29.232.224.85,29.191.192.197,29.119.83.12`

**B 的 IP**:`29.191.210.123,29.127.64.8,29.127.65.89,29.119.84.187,29.127.82.110,29.191.209.108,29.119.97.119,29.119.99.50`

### Rank 分配结果

```
机器 A (NODE_RANK_OFFSET=0):  global rank 0  1  2  3  4  5  6  7
机器 B (NODE_RANK_OFFSET=8):  global rank 8  9 10 11 12 13 14 15
                              → 0..15 完整无重复
```

---

## 三、训练超参 (A/B 必须完全一致)

| 参数 | 值 | 说明 |
|---|---|---|
| 数据 | 全量 27500 episode | 不设 TASK/EPISODE 过滤 → 全 null → 全量 |
| `OPTIMIZER_LR` | `2e-4` | |
| `MAX_SAMPLES_PER_BATCH` | `16` | 每卡每 step 样本数 |
| `GLOBAL_BATCH` | `2048` | = 16 node × 8 gpu × 16 = 2048 (脚本自动算) |
| `MAX_ITER` | `3000` | ≈ 1.09 epoch (5.64M 窗口 / gbs2048 ≈ 2752 step/epoch) |
| `SAVE_ITER` | `500` | 每 500 step 存一次 checkpoint |
| `ROBOTWIN_DATASET_REPEAT` | `1` | 全量无需 repeat |
| 分片 | `shard_map_style_dataset=true` | DistributedSampler 各 rank 看不重叠的 1/128 |
| shuffle | `+...shuffle=true` | 圈内打散 |

> 想训两遍数据 (~2.18 epoch) 把 **A 和 B 的 `MAX_ITER` 同时改成 6000**。
> 任何超参改动两边都要同步,否则 job 不对齐。

---

## 四、关键机制说明

### 4.1 `NODE_RANK_OFFSET` (多 head 启动的核心)

`run_pdsh_robotwin_cosmos3nano.sh` 支持 `NODE_RANK_OFFSET`(默认 0,单 head 行为不变):
- 每个 head 只 pdsh 到**自己** `EXP_IP_LIST` 里的节点
- 节点的 global rank = `NODE_RANK_OFFSET + 本地下标`
- 这样两个 head 各启 8 节点,rank 不冲突,torchrun 静态 rendezvous 在 `MASTER_ADDR:MASTER_PORT` 等齐 16 个节点后开始

### 4.2 wandb 聚成一个 run

两 head 是同一个 job,`WANDB_NAME` 用了**固定值** `${EXP_TAG}_0624`(不带 `$(date)`),A/B 一致 → 在 wandb 聚成同一个 run。
- `WANDB_GROUP=robotwin_all_task`
- `EXP_TAG=all_task_lr2e-4_gbs2048_16node`

### 4.3 日志

启动脚本会把 head 侧所有输出 (pdsh 过程 / `Starting node N` / 报错) 同时打到终端并存盘:

```
/mnt/lyn/wjh/workspace/cosmos3/wam/logs/<EXP_TAG>_A_<时间戳>.log   # 机器 A
/mnt/lyn/wjh/workspace/cosmos3/wam/logs/<EXP_TAG>_B_<时间戳>.log   # 机器 B
```

各节点各 rank 的**训练 log**(loss / iter_speed)在:
```
$OUTPUT_ROOT/logs/robotwin_lerobot_action_sft_nano_n16_rank{0..15}.log
# OUTPUT_ROOT = OUTPUT_BASE_ROOT/TRAINING_NAME = .../all_task_lr2e-4_gbs2048_16node
```

---

## 五、跨机网络 (NCCL)

当前默认走 **IB** (`NCCL_IB_DISABLE=0`)。实测 16node 比单 8node 更快 (22s vs 50s/step),说明 IB 互联正常。

若跨机 NCCL 在初始化处 **hang 住** (卡着不动、不报错),说明两机 IB fabric 不互通,回退**纯 TCP**:在**两台机器**启动前都 export:

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=<两机互通的网卡名>   # 用 `ip addr` 确认两机互通网段, 默认 bond1 不一定跨机通
```

---

## 六、常见坑

1. **必须先 A 后 B**,且 B 的 `MASTER_ADDR` 指向 A 的 rank0,不是 B 自己。
2. **A/B 超参必须一致** (lr / gbs / max_iter / wandb_name),改一个要两边同步改。
3. 机器 A 看到 `WARN: this head drives only 8 of 16 global nodes` 是**预期提示**,不是错误。
4. 各 head 只需能 pdsh 到**自己**那 8 个节点,不需要跨机 pdsh。
5. 首步 (iteration 1) 会很慢 (~280s,含 CUDA 编译 + NCCL 建链),之后稳定到 ~22s/step,属正常。
