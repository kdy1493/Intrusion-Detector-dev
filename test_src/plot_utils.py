"""plot_utils – 실시간 시각화

────────────────────────────────────────────
● 역할
    ▸ CADA_process 가 Queue 에 푸시한 CSIResult 를
      Matplotlib 로 실시간 Subplot 출력

● 주요 함수
    run_plotter_universal(topics:list[str],
                          result_queue:Queue,
                          buffer_size:int = 4096,
                          refresh_ms:int = 1000)

● 동작
    ▸ 토픽별로 _LiveBuffer(deque) 유지
    ▸ FuncAnimation 주기마다 Queue → 버퍼 drain
    ▸ 토픽 수가 1개면 1×1, 그 외엔 2열 그리드 자동 배치
    ▸ Plot 창 종료 시 plt.close('all') + time.sleep(0.5) → Tk 경고 최소화

● 커스터마이징 포인트
    ▸ cols/rows 계산식, figsize 배수, refresh_ms 값
────────────────────────────────────────────
"""

import time
from collections import deque
from multiprocessing import Queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class _LiveBuffer:
    """토픽별 실시간 표시에 필요한 순환 버퍼 관리"""
    def __init__(self, maxlen=4096):
        self.ts   = deque(maxlen=maxlen)  # timestamp
        self.val  = deque(maxlen=maxlen)  # activity_detection
        self.flag = deque(maxlen=maxlen)  # activity_flag (0/1)
        self.th   = deque(maxlen=maxlen)  # threshold


def run_plotter_universal(topics, result_queue: Queue,
                          buffer_size: int = 4096,
                          refresh_ms: int = 1000):
    """
    `CADA_process.py`가 넘겨주는 Queue(CSIResult 객체)에서 데이터를 꺼내
    실시간으로 subplot에 그립니다.

    Parameters
    ----------
    topics : list[str]
        ESP-MAC / MQTT topic 리스트. subplot 순서를 결정.
    result_queue : multiprocessing.Queue
        CADA_process 에서 put 해주는 CSIResult 객체 수신용.
    buffer_size : int, optional
        토픽별 순환 버퍼 길이(표시할 최대 프레임 수).
    refresh_ms : int, optional
        matplotlib FuncAnimation 새로고침 주기(ms).
    """
    # ── 1. 토픽별 버퍼 초기화 ───────────────────────────────────────
    buffers = {t: _LiveBuffer(maxlen=buffer_size) for t in topics}

    # ── 2. Queue consumer: non-blocking pull ─────────────────────
    def _drain_queue():
        """큐에 쌓인 모든 항목을 버퍼에 push (빈 큐일 때까지)"""
        while True:
            try:
                item = result_queue.get_nowait()
                buf = buffers[item.topic]
                buf.ts.append(item.timestamp)
                buf.val.append(item.activity_detection)
                buf.flag.append(item.activity_flag)
                buf.th.append(item.threshold)
            except Exception:
                break   # Empty → 바로 반환

    # ── 3. 애니메이션 콜백 ───────────────────────────────────────
    cols = 1 if len(topics) == 1 else 2          # 토픽이 1개면 1열, 그 외 2열
    rows = (len(topics) + cols - 1) // cols      # 필요한 행 수
    fig, axes = plt.subplots(
        nrows=rows, 
        ncols=cols,
        figsize=(6 * cols, 3 * rows),
        squeeze=False
    )
    plt.tight_layout(pad=2.0)

    def _update(_frame_idx):
        _drain_queue()                       # 새 데이터 반영
        for ax in axes.flat:                 # 축 초기화
            ax.cla()
        for idx, topic in enumerate(topics):
            r = idx // 2
            c = idx % 2
            ax = axes[r][c]
            buf = buffers[topic]
            if not buf.ts:
                ax.set_title(f"{topic} (no data)")
                continue
            ax.plot(buf.ts, buf.val, label="ActivityDetection")
            ax.step(buf.ts, buf.flag, label="ActivityFlag", where="mid", color="r")
            ax.plot(buf.ts, buf.th, label="Th", linestyle="--", color="g")
            ax.set_title(topic)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(loc="upper left")
        fig.autofmt_xdate()

    # ── 4. 애니메이션 시작 ───────────────────────────────────────
    ani = FuncAnimation(fig, _update, interval=refresh_ms, cache_frame_data=False)
    plt.show()
    _exit()

def _exit(_sig=None, _frm=None):
    """
    Tk 가 남기는 'main thread is not in main loop' traceback을
    확실히 없애기 위해 파이썬 정리 단계를 건너뛰고
    C 레벨에서 바로 프로세스를 종료한다.
    """
    try:
        import matplotlib.pyplot as _plt
        _plt.close('all')
    except Exception:
        pass
    os._exit(0)

