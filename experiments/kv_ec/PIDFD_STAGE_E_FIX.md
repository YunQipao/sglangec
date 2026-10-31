# Stage E pidfd Python compatibility fix

本次只修复 pidfd 平台适配、清理身份复核和启动错误传播。不修改 GPU Restore、
XOR、RS、RDMA、端口配置、Shared GPU 策略或运行环境；不操作现有 RS 服务。

## 实现与安全边界

`pidfd_compat.py` 集中提供 open/send/probe。优先 Python 原生绑定；缺失时仅在
Linux 使用 libc 同名封装（ctypes 明确参数和返回类型、use_errno=True）。不调用
裸 syscall，不猜 CPU syscall 编号；libc 负责当前 CPU ABI。libc 符号缺失或
内核/权限不支持则拒绝启动。原生绑定存在但调用报错时也不绕过错误换路径。
EBADF、ESRCH、ENOSYS、EPERM 等 errno 原样保留；只有进程已消失可按退出处理。
没有 os.kill/killpg/pkill 降级。

启动探测和清理发送信号使用同一兼容层。preflight 每节点探测；即使绕过正常
benchmark preflight 直接运行 trial，也会在 prepare/init-trial/服务启动前，
在两个 Host 的目标 Python 下再次探测，输出保存到当前 Trial 的
`pidfd-preflight.log`。

仍逐成员检查原 session、UID、boot_id、leader start_ticks 和 Trial 环境 token；
打开 pidfd 后，在发送前后核验 PID、start_ticks、boot_id、UID、PGID、session/SID
和 run_id。pidfd 绑定进程实例，避免数字 PID 复用误杀。进程退出后 /proc 消失可
忽略；environ 权限错误最多重试 0.5s，每次重新验证身份，仅身份一致的 Z/X
进程允许结束复核，活进程无法验证则拒绝。PID 改变或存活进程权限错误均不忽略。

清理语义保持 SIGINT/30s、SIGTERM/10s、SIGKILL/5s 和最终 wait/5s；不扩大清理
范围。唯一额外等待是退出时身份复核的有界重试。Supervisor 在创建服务目录前
失败，也在已认证的 Trial 下创建服务目录并保存 `exit.json` 的异常和 traceback；
wait-file 每 0.5s 读取退出记录并报告实际错误，正常等待上限仍为 900s。外层
Supervisor stdout/stderr 已由 coordinator 保存为 Trial 内 `<name>-nodeN.log`。
此修复处理可捕获的早期异常；SIGKILL/断电等无法写退出记录的情况仍可能等待超时。

## 现场能力与测试

当前执行环境 hostname=disai-4090-3，x86_64；无法据此宣称是 232/233 本体。
指定 Conda Python 3.12.15 的两项 Python 绑定都缺失，libc 自身信号 0 实测成功。
系统 Python 3.12.3 两项原生绑定均存在。233 SSH 只读探测尝试被沙箱以
`socket: Operation not permitted` 拒绝；232/233 的现场版本仍须以下命令确认。

最终结果：Conda 四个测试模块共 148 项，147 通过、1 跳过（TCP bind 被沙箱
禁止），0 失败；系统 Python 的新增专项模块 12/12 通过。系统 Python 额外跑
test_perf_ports 时已有 RS mock 导入测试报 AttributeError（remote_rs_client
不可导入），因此不声称系统 Python 的该模块通过；指定 Conda 环境该测试通过。
首轮退出竞态及测试重复写证据问题已修复并在最终回归通过。

单机回归命令（在仓库根目录）：

```bash
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
"$PY" -m unittest experiments.kv_ec.test_pidfd_compat \
  experiments.kv_ec.test_perf_ports experiments.kv_ec.test_perf_stage_e \
  experiments.kv_ec.test_tp4_stage_e
/usr/bin/python3 -m unittest experiments.kv_ec.test_pidfd_compat
```

新增测试覆盖原生优先、真实 libc 缺失绑定路径、缺失 libc/非 Linux/ENOSYS 快速
拒绝、各 errno、原生权限失败不降级、两节点探测在服务启动前完成、普通进程和
SSH 隧道 CPU 替身清理、发送前后身份变化拒绝、Trial token 拒绝、Supervisor
早期异常立即反馈、三轮 Re-Prefill 隧道 CPU 替身、三轮 Two-level 角色 CPU 服务
及真实 Port Guard/锁获取释放。已有测试覆盖 PID 复用、其他 UID、不属本 Trial
的 session、leader 退出后新 PGID 子进程。TCP listener 实测被沙箱禁止时跳过，
不能视为端口真实 bind 回归通过。以上均不是真实双节点或 GPU/RDMA 实验证据。

## 232/233 只读能力检查

在有网络权限的终端运行。这里仅发送信号 0 到探针自身，不向现有服务发信号。
第一段不依赖修复文件，可在同步前运行；第二段在同步后检查实际选用后端。

```bash
for host in iic@10.102.0.232 iic@10.102.0.233; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    /media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python - <<'PY'
import ctypes, os, platform, signal, sys
print(platform.node(), platform.machine(), sys.version)
print('os.pidfd_open=', hasattr(os, 'pidfd_open'))
print('signal.pidfd_send_signal=', hasattr(signal, 'pidfd_send_signal'))
libc = ctypes.CDLL(None, use_errno=True)
op = libc.pidfd_open
op.argtypes, op.restype = (ctypes.c_int, ctypes.c_uint), ctypes.c_int
send = libc.pidfd_send_signal
send.argtypes, send.restype = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint), ctypes.c_int
fd = op(os.getpid(), 0)
if fd < 0:
    e = ctypes.get_errno()
    raise OSError(e, os.strerror(e))
try:
    if send(fd, 0, None, 0) < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    print('libc/kernel pidfd signal-zero supported')
finally:
    os.close(fd)
PY
done
for host in iic@10.102.0.232 iic@10.102.0.233; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    'cd /media/sda1/iic_data/luotao/sglangec/sglang && /media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python -m experiments.kv_ec.perf_stage_e check-pidfd'
done
```

## 待执行的真实双节点验证命令

本次没有执行以下实验，不标记 PASS。先由操作者审核 diff、同步清单并在两端
核验 SHA256；本次没有自动覆盖 233 源文件。配置使用当前实际 TP=2/shared
配置的绝对路径，不使用 example 覆盖 GID/端口/环境。命令在 232 仓库根运行：

```bash
set -euo pipefail
cd /media/sda1/iic_data/luotao/sglangec/sglang
PY=/media/sda1/iic_data/anaconda3/envs/sglang-ec/bin/python
export CONFIG=/absolute/path/to/current-shared-tp2-config.json
"$PY" - <<'PY'
import json, os
from experiments.kv_ec.perf_preflight import validate_config
c = json.load(open(os.environ['CONFIG']))
validate_config(c)
assert c['tp_size'] == 2 and c['gpu_admission']['mode'] == 'shared'
assert c['rs'] == [2, 1]
PY
RUN=$(mktemp -d /tmp/stage-e-pidfd-validation.XXXXXXXX)
export RUN
printf '%s\n' "$RUN"
# 2 个 smoke + 各 3 轮 = 总计 8 轮；预检保留现有 RS 并验证累计容量。
"$PY" -m experiments.kv_ec.perf_benchmark preflight \
  --config "$CONFIG" --output "$RUN/preflight-eight.json" --repeat 8
"$PY" - <<'PY'
import json, os
r = json.load(open(os.environ['RUN'] + '/preflight-eight.json'))
assert r['status'] == 'PASS', r
PY
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/two-level-smoke" --repeat 1 --recovery-mode two-level
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-smoke" --repeat 1 --recovery-mode reprefill
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/two-level-r3" --repeat 3 --recovery-mode two-level
"$PY" -m experiments.kv_ec.perf_benchmark repeat \
  --config "$CONFIG" --output "$RUN/reprefill-r3" --repeat 3 --recovery-mode reprefill
"$PY" -m experiments.kv_ec.perf_benchmark collect \
  --roots "$RUN/two-level-smoke" "$RUN/reprefill-smoke" \
    "$RUN/two-level-r3" "$RUN/reprefill-r3" --output "$RUN/summary.json"
```

每个 Trial 应审查 `cleanup.json`、两端 `port-release.json`、`port-check.json`、
各服务 `process-cleanup.json`，Re-Prefill 还需 `tunnel/process-cleanup.json`，
远端证据在 `json-evidence-233.zip`。cleanup PASS
要求剩余成员为空、端口释放；lock 文件会保留，释放的是 flock，不能用文件存在
判定锁泄漏。RS epoch 前后保持一致且 health READY。任何失败都保留当前结果，
下一次使用新输出目录；不要重启 RS 或移除历史结果以通过验证。
