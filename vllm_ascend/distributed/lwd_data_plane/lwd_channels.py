# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UP and DOWN channels of the LWD edge-cloud data plane.

Owns the physical channels (UP edge->cloud, DOWN cloud->edge):
  * channel HCCL communicators: two dedicated ``new_group`` groups per DP
    connection, over its endpoint pair (``LwdChannelEndpoints``: the first
    edge rank and the first cloud rank);
  * one NPU stream per channel of each DP connection the local rank
    endpoints (channel ops are bridged onto it, keeping the compute stream
    free);
  * init-time warmup of every connection's channels in a fixed global
    order (moves the first-use HCCL rendezvous out of the pipeline, where
    the two sides could otherwise arrive on different channels and
    deadlock).

In a single_instance topology the edge process serves every DP
connection, so its endpoint rank holds the channel pairs of all of them;
a cloud endpoint rank holds exactly its own DP's pair.

Design: docs/design/lwd_data_plane.md in the vLLM repository.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.distributed.lwd import LwdChannelEndpoints, LwdParallelGroupsSpec
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

# (dp_idx, channel) -> (device_group, peer_global_rank), endpoint ranks only
_LWD_CHANNEL_GROUPS: dict[tuple[int, LwdChannelType], tuple[dist.ProcessGroup, int]] = {}
_LWD_CHANNEL_STREAMS: dict[tuple[int, LwdChannelType], torch.npu.Stream] = {}
_LWD_ENDPOINTS: tuple[LwdChannelEndpoints, ...] = ()
_LWD_CHANNELS_INITIALIZED = False


def lwd_init_channels() -> None:
    """Create the UP and DOWN channels of every DP connection. Idempotent.

    Called once by every worker of the shared world after its distributed
    environment is up. Group creation is a world collective, so every rank
    creates the groups of every connection in the order of
    ``LwdParallelGroupsSpec``; only the two endpoint ranks keep them.
    """
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    if _LWD_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    assert lwd_config is not None, "the data plane requires lwd_config"
    spec = LwdParallelGroupsSpec.from_topology(lwd_config.topology)
    backend = dist.get_backend(get_world_group().device_group)
    my_rank = dist.get_rank()
    for endpoints in spec.channel_endpoints:
        ranks = [endpoints.edge_rank, endpoints.cloud_rank]
        for channel in LwdChannelType:
            # Both channels span the same rank pair; the direction is given
            # by who sends.
            group = dist.new_group(ranks, backend=backend)
            if my_rank in ranks:
                peer = endpoints.cloud_rank if my_rank == endpoints.edge_rank else endpoints.edge_rank
                _LWD_CHANNEL_GROUPS[(endpoints.dp_idx, channel)] = (group, peer)
                _LWD_CHANNEL_STREAMS[(endpoints.dp_idx, channel)] = torch.npu.Stream()
    _LWD_ENDPOINTS = spec.channel_endpoints_of(my_rank)
    # The cloud-internal embeds broadcast does not build its own group: a
    # dedicated new_group delivered nothing under torch_npu in testing
    # (suspected group creation order mismatch), so it uses the framework TP
    # device_group, whose creation order the framework guarantees. The
    # broadcast is issued on the compute stream at injection time, FIFO with
    # the forward TP collectives, so the two cannot interleave.
    logger.info(
        "[LWD] data-plane channels created for %s (my_rank=%d, endpoints=%s)",
        spec.channel_endpoints,
        my_rank,
        _LWD_ENDPOINTS,
    )
    # Every connection's two channels, always warmed up: the first-use HCCL
    # rendezvous is moved to init time, where both sides are guaranteed to
    # arrive in the same fixed order.
    lwd_warm_up_channels()
    _LWD_CHANNELS_INITIALIZED = True


def _lwd_wait_with_timeout(work: dist.Work, timeout_s: float, what: str) -> None:
    """等待一个 torch Work 完成;超时抛 TimeoutError,HCCL 异常原样抛出。

    ``Work.wait()`` 没有 timeout 参数,因此用"守护线程 + Event"实现:线程只
    负责 wait 并把异常回传,主线程负责超时判定。超时后raise。
    """
    done = threading.Event()
    errors: list[BaseException] = []

    def _waiter() -> None:
        try:
            work.wait()
        except BaseException as exc:  # noqa: BLE001 - 通信错误必须回传
            errors.append(exc)
        finally:
            done.set()

    threading.Thread(target=_waiter, name="lwd-hs", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"[Lwd][handshake] timeout {timeout_s:.0f}s: {what}")
    if errors:
        raise RuntimeError(f"[Lwd][handshake] failed: {what}: {errors[0]!r}") from errors[0]


def _lwd_echo_handshake(engine_index: int, edge_rank: int, cloud_rank: int, timeout_s: float) -> dict[str, float]:
    """单个 engine pair 的 echo 往返:边发已知值 → 云原样回 → 边逐位比对。

    判据只在边侧 recv:HCCL ``isend`` 完成只表示本地发送缓冲可复用,
    不代表对端已收到。云侧超时取 ``2x``,用于边侧失败后的对端自清。


    """
    magic = 0x4C5744  # "LWD"
    payload_dtype = torch.int32
    payload_elems = 3
    my_rank = dist.get_rank()
    up_group, up_peer = _LWD_CHANNEL_GROUPS[(engine_index, LwdChannelType.UP)]
    down_group, down_peer = _LWD_CHANNEL_GROUPS[(engine_index, LwdChannelType.DOWN)]
    t0 = time.monotonic()
    marks: dict[str, float] = {}
    # 可变容器:主线程写阶段,看门狗线程读(只用于日志,允许轻微竞争)
    phase = ["post"]
    budget_s = timeout_s if my_rank == edge_rank else timeout_s * 2
    finished = threading.Event()

    """HCCL 建链会阻塞在 ``isend``/``irecv`` 或``.cpu()`` 同步里,
    而 ``HCCL_CONNECT_TIMEOUT``的合法下限是 120s, 无法用它压到 30s。
    看门狗到期先打一条 ``FAILED phase=… (watchdog …)`` 然后 ``os._exit(1)``;
    它不调用任何 NPU API, 因此不影响流语义。"""

    def _watchdog() -> None:
        if finished.wait(budget_s):
            return
        logger.error(
            "[Lwd][handshake] engine=%d rank=%d FAILED phase=%s "
            "(watchdog %.0fs, blocked in HCCL) up_peer=%d down_peer=%d",
            engine_index,
            my_rank,
            phase[0],
            budget_s,
            up_peer,
            down_peer,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    threading.Thread(target=_watchdog, name="lwd-wd", daemon=True).start()

    def _mark(name: str) -> None:
        marks[name] = (time.monotonic() - t0) * 1000.0

    def _fmt(value: float | None) -> str:
        """阶段耗时格式化：未到达的阶段打印 "-"。"""
        return "-" if value is None else f"{value:.1f}"

    try:
        if my_rank == edge_rank:
            send = torch.tensor(
                [magic, engine_index, 0],
                dtype=payload_dtype,
                device="npu",
            )
            recv = torch.zeros(payload_elems, dtype=payload_dtype, device="npu")
            send_work = dist.isend(send, dst=up_peer, group=up_group)
            recv_work = dist.irecv(recv, src=down_peer, group=down_group)
            _mark("post")
            phase[0] = "recv"
            _lwd_wait_with_timeout(
                recv_work,
                timeout_s,
                f"engine={engine_index} dir=DOWN peer={down_peer} rank={my_rank}",
            )
            _mark("recv")
            phase[0] = "send"
            _lwd_wait_with_timeout(send_work, timeout_s, f"engine={engine_index} dir=UP-send")
            _mark("send")
            phase[0] = "verify"
            if not torch.equal(recv.cpu(), send.cpu()):
                raise RuntimeError(
                    f"[Lwd][handshake] content mismatch engine={engine_index} "
                    f"expect={send.tolist()} got={recv.tolist()}"
                )
            _mark("verify")
        else:
            recv = torch.zeros(payload_elems, dtype=payload_dtype, device="npu")
            recv_work = dist.irecv(recv, src=up_peer, group=up_group)
            _mark("post")
            phase[0] = "recv"
            _lwd_wait_with_timeout(
                recv_work,
                timeout_s * 2,
                f"engine={engine_index} dir=UP peer={up_peer} rank={my_rank}",
            )
            _mark("recv")
            phase[0] = "verify"
            got = [int(v) for v in recv.cpu().tolist()]
            if got[0] != magic or got[1] != engine_index:
                raise RuntimeError(f"[Lwd][handshake] bad payload engine={engine_index} got={got}")
            _mark("verify")
            phase[0] = "send"
            send_work = dist.isend(recv, dst=down_peer, group=down_group)
            _lwd_wait_with_timeout(
                send_work,
                timeout_s * 2,
                f"engine={engine_index} dir=DOWN-send peer={down_peer}",
            )
            _mark("send")
    except BaseException:
        logger.error(
            "[Lwd][handshake] engine=%d FAILED phase=%s post=%s recv=%s send=%s verify=%s (ms)",
            engine_index,
            phase[0],
            _fmt(marks.get("post")),
            _fmt(marks.get("recv")),
            _fmt(marks.get("send")),
            _fmt(marks.get("verify")),
        )
        raise
    finally:
        finished.set()
    logger.debug(
        "[Lwd][handshake] engine=%d round-trip=%.2fms connect=%.2fms",
        engine_index,
        (time.monotonic() - t0) * 1000,
        marks.get("post", 0.0),
    )
    return marks


def lwd_warm_up_channels() -> None:
    """Verify UP/DOWN round trips in topology order before serving requests."""
    timeout_s = float(ascend_envs.VLLM_ASCEND_LWD_HCCL_HANDSHAKE_TIMEOUT_S)
    if not 0 < timeout_s < float("inf"):
        raise ValueError("LWD HCCL handshake timeout must be finite and positive")
    my_rank = dist.get_rank()
    started = time.monotonic()
    for endpoints in _LWD_ENDPOINTS:
        _lwd_echo_handshake(endpoints.dp_idx, endpoints.edge_rank, endpoints.cloud_rank, timeout_s)
    echo_ms = (time.monotonic() - started) * 1000
    barrier_start = time.monotonic()
    get_world_group().barrier()
    logger.info(
        "[LWD][handshake] OK connections=%d rank=%d echo=%.1fms barrier=%.1fms",
        len(_LWD_ENDPOINTS),
        my_rank,
        echo_ms,
        (time.monotonic() - barrier_start) * 1000,
    )


def lwd_channels_initialized() -> bool:
    return _LWD_CHANNELS_INITIALIZED


def _lwd_channel(dp_idx: int, channel: LwdChannelType) -> tuple[dist.ProcessGroup, int]:
    try:
        return _LWD_CHANNEL_GROUPS[(dp_idx, channel)]
    except KeyError:
        raise RuntimeError(
            f"[LWD] rank {dist.get_rank()} holds no {channel.value} channel of "
            f"dp_idx {dp_idx}; only the endpoint ranks of that DP connection do"
        ) from None


def lwd_get_channel_device_group(channel: LwdChannelType, dp_idx: int = 0) -> dist.ProcessGroup:
    group, _ = _lwd_channel(dp_idx, channel)
    return group


def lwd_get_channel_peer(channel: LwdChannelType, dp_idx: int = 0) -> int:
    """Global rank of this process's P2P peer on the DP connection's channel."""
    _, peer = _lwd_channel(dp_idx, channel)
    return peer


def lwd_get_channel_stream(channel: LwdChannelType, dp_idx: int = 0) -> torch.npu.Stream:
    _lwd_channel(dp_idx, channel)
    return _LWD_CHANNEL_STREAMS[(dp_idx, channel)]


def lwd_destroy_channels() -> None:
    global _LWD_CHANNELS_INITIALIZED, _LWD_ENDPOINTS
    for (dp_idx, channel), (group, _) in list(_LWD_CHANNEL_GROUPS.items()):
        try:
            dist.destroy_process_group(group)
        except Exception:
            logger.warning("[LWD] failed to destroy the data-plane group for dp%d %s", dp_idx, channel.value)
    _LWD_CHANNEL_GROUPS.clear()
    _LWD_CHANNEL_STREAMS.clear()
    _LWD_ENDPOINTS = ()
    _LWD_CHANNELS_INITIALIZED = False
