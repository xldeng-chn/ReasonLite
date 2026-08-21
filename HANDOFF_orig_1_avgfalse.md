# orig_1_avgfalse 续训 Handoff

## 参数

本文档中所有账户相关路径用 `<USER>` 占位。接手者把 `<USER>` 替换为自己的 GPFS 账户名即可复用全部路径（示例值 `dengxianglong`）。需替换的位置：

- `/user/<USER>/...` —— GPFS 账户根及派生路径（outputs / backups / scripts / cache）
- `devspace-<USER>-neimeng-devbox-612515` —— teleport devbox 主机名
- `~/.tsh/keys/teleport.cybertron.modelbest.co/<USER>` 与 `.../<USER>-ssh/teleport.cybertron.modelbest.co-cert.pub` —— teleport 证书/私钥路径

`xldeng-chn` 是 DevCloud 组织名（非用户名），保持不变。

## 目标

跑完 stage1 全程（135,440 步），与 packed 侧 job 574469 做"同起点同终点"对比。续训从 checkpoint-10400 恢复，跑到 135,440 自然结束。**已完成（Aug 18）。**

## 训练配置（SSOT：`ReasonLite/.claude/worktrees/parity-baseline/train/config_stage1.yaml`）

| 项 | 值 |
|---|---|
| 模型 | Qwen3-0.6B，bf16，flash_attention_2 |
| 数据 | reason_lite-dataset split medium，dataset_num_proc 96 |
| 并行 | 2 节点 × 8 = 16×H100 |
| per_device_train_batch_size | 16 |
| gradient_accumulation_steps | 1 |
| **global batch size** | **256**（16 GPU × 16 × 1） |
| max_steps | 135,440（num_train_epochs 8 推导） |
| seed | 42 |
| average_tokens_across_devices | false（基线侧，匹配 packed 对照） |
| save_total_limit | 30（滚动窗口） |
| 资源 | priority NORMAL，cpu 105 / memory 1207 每节点，池 aiforai H100 |

## 代码栈与依赖

| 组件 | 来源 | 落点 |
|---|---|---|
| ReasonLite | cctl `--code-type git` 从 DevCloud 拉取，`--git-ref parity-baseline` | 挂载到 `/local/apps/ReasonLite` |
| open-r1 | `launch_h100.sh` **运行时**从 DevCloud **HTTPS:443** 克隆（`OPENR1_GIT_REPO` + 可选 `OPENR1_GIT_USER`/`OPENR1_GIT_TOKEN`，见 `setup_env.sh`） | `/local/app/open-r1`，再 `pip install --no-deps -e` |
| Python 依赖 | PyPI（清华镜像），`requirements_train.txt` | 全 PyPI，无本地包 |

DevCloud URL（SSOT 在 `train/setup_env.sh`）：
- ReasonLite: `git@codehub.devcloud.cn-north-4.huaweicloud.com:66cb35255b8140c08f7af25e4a10542d/xldeng-chn/ReasonLite.git`（cctl 提交侧克隆，节点只拿挂载结果）
- open-r1:    `https://codehub.devcloud.cn-north-4.huaweicloud.com/66cb35255b8140c08f7af25e4a10542d/xldeng-chn/open-r1.git`（节点运行时克隆）

**已知坑：训练节点到 DevCloud:22 不通**。SSH 克隆（`git@codehub...`）在训练节点上 `Connection timed out`（smoke 747104 实测）——节点没有到 DevCloud 端口 22 的网络路由。故 open-r1 改走 HTTPS:443。私有仓库需凭据：通过 cctl `--env OPENR1_GIT_USER=... --env OPENR1_GIT_TOKEN=...` 注入（token 是密钥，**不要写入仓库**）；`launch_h100.sh` 在 URL 里拼 `https://<user>:<token>@host/...`。无 token 时 `GIT_TERMINAL_PROMPT=0` 让克隆在认证挑战处快速失败而非挂起，可作 :443 可达性探针。

**本地包依赖核查**：`requirements_train.txt` 全 PyPI 无本地包；唯一依赖本地包的 pip 安装是 open-r1 editable（`pip install --no-deps -e ${OPENR1_ROOT}`），由运行时克隆满足。FA3 egg（`launch_h100.sh` `${REASONLITE_WORKSPACE_ROOT}/wheels/flash_attn_3-...egg`）是本地 GPFS 制品，仅 `flash_attention_3` 分支触发，本分支用 FA2，不触发。

## 关键路径

| 用途 | 路径 |
|---|---|
| 输出目录（续写） | `/user/<USER>/outputs/parity/20260807T070952Z_orig_1_avgfalse/output/` |
| 续训日志 | `.../output_resume_full.log` |
| 断点 | `.../output/checkpoint-10400/` |
| 最终模型 | `.../output/checkpoint-135440/`（末步完整存档） |
| 备份目录 | `/user/<USER>/backups/` |
| guard 脚本 | `/user/<USER>/scripts/backup_10k_ckpts.sh` |
| guard daemon | `/user/<USER>/scripts/backup_10k_daemon.sh` |
| guard 日志 | `/user/<USER>/scripts/backup_10k_ckpts.log` |
| 心跳 | `/user/<USER>/scripts/backup_10k_heartbeat` |

## 已知坑：HF Trainer resume 覆盖 CLI `--save_steps`

CLI 传 `--save_steps 10000` 在解析阶段生效，但 `_load_from_checkpoint`（transformers 4.52.3, trainer.py:2884）从被恢复 checkpoint 的 `training_args.bin` 加载原存值（100），覆盖当前 args。**结果：实际仍每 100 步存一次**，不是每 10k。

## 缓解方案（方案 A，已生效全程）

不中断训练，改用运维守护脚本在每个 10k 边界 checkpoint 被滚动窗口删除前自动备份：

- `backup_10k_ckpts.sh`：扫描 30000–135440 各 10k 边界，目录 mtime 静默 300s 后 `cp -a` 到 `backups/checkpoint-${step}_orig_1_avgfalse_full`，`.DONE` 标记幂等。
- `backup_10k_daemon.sh`：`flock -n /tmp/backup_10k.lock` + 120s 循环（容器无 cron）。

## 进展（终态，Aug 18）

- **135,440 / 135,440 完成**，stage1 跑完，最终 `checkpoint-135440` 落盘（`global_step=135440=max_steps`）。
- 速度 ~6.3 s/it → 10k 步 ≈ 16.4h。
- **已备份 13 个边界**（全部 9.3G，global_step 精确命中，guard 日志可追溯）：30000 / 40000 / 50000 / 60000 / 70000 / 80000 / 90000 / 100000 / 110000 / 120000 / 130000 / 135440。
- guard daemon 心跳正常。

## 访问方式

devbox 通过 teleport SSH（证书有效期约 16h，过期需重新 `tsh login`）：

```
tsh --proxy=teleport.cybertron.modelbest.co:443 login teleport.cybertron.modelbest.co
ssh -o ProxyCommand="tsh proxy ssh --cluster=teleport.cybertron.modelbest.co --proxy=teleport.cybertron.modelbest.co:443 %r@%h:%p" \
    -o IdentityFile="$HOME/.tsh/keys/teleport.cybertron.modelbest.co/<USER>" \
    -o CertificateFile="$HOME/.tsh/keys/teleport.cybertron.modelbest.co/<USER>-ssh/teleport.cybertron.modelbest.co-cert.pub" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    root@devspace-<USER>-neimeng-devbox-612515
```

## 提交命令（参考）

```bash
D=/user/<USER>/outputs/parity/20260807T070952Z_orig_1_avgfalse
cctl pytorchjob create \
  --project neimeng-devbox --cluster paratera_train --resource-pool aiforai \
  --billing-account-id N00007 --image infra/nvidia-pytorch:latest \
  --code-type git --git-path git@codehub.devcloud.cn-north-4.huaweicloud.com:66cb35255b8140c08f7af25e4a10542d/xldeng-chn/ReasonLite.git \
  --git-ref parity-baseline \
  --gpu 8 --gpu-model H100 --nodes 2 --cpu 105 --memory 1207 --priority NORMAL \
  --env OPEN_R1_DISABLE_AUTO_RESUME=1 \
  --entry "set -o pipefail; mkdir -p $D && REASONLITE_EXTRA_ARGS='--output_dir $D/output --resume_from_checkpoint $D/output/checkpoint-10400 --save_steps 10000' bash /local/apps/ReasonLite/train/launch_h100.sh full stage1 2>&1 | tee $D/output_resume_full.log"
```

## 验证判据

- [x] 日志 `Total optimization steps = 135,440`
- [x] 从 10400 续接，非重来
- [x] 各 10k 边界 checkpoint 已备份（guard 自动，13 个边界无一遗漏）
- [x] 进度 135440/135440，自然结束，最终模型落盘
