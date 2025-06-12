"""
plot_utils.py
+----
Wi-Fi CSI 데이터 시각화 유틸리티 모듈.

주요 기능
+----
• convert_csi_to_amplitude 함수: I/Q 값 진폭 변환 기능.
• plot_csi_amplitude, plot_csi_amplitude_from_file 함수: 오프라인 진폭 그래프 출력 기능.
• plot_realtime_universal 함수: 실시간 특징 버퍼 통합 시각화 기능.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates


def convert_csi_to_amplitude(file_path, SUBCARRIER_NUM=52):  # src/utils.py
    """
    Convert CSI data to amplitude.
    :param file_path: Path to the CSV file containing CSI data.
    :param SUBCARRIER_NUM: Number of subcarriers (default = 52)

    Example usage:
    NO_ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_NoActivity\merged_csi_data_noactivity.csv"
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH, SUBCARRIER_NUM) # amp: signal amplitude, ts: timestamp
    :return: Tuple of amplitude and timestamp arrays.
    """
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M.%S")

    I = df[[f"I{i}" for i in range(SUBCARRIER_NUM)]].values
    Q = df[[f"Q{i}" for i in range(SUBCARRIER_NUM)]].values
    amp = np.sqrt(I**2 + Q**2)
    ts = ts = df["timestamp"]

    return amp, ts

def plot_csi_amplitude(amp, time_stamp, title="None", FRAME_NUM=500,
                       amp2=None,amp3=None ):  # in src/utils.py
    """
    Loads CSI data, calculates amplitude, and plots it.
    param amp : amplitude
    param time_stamp : timestamp
    param title : title of the plot
    param FRAME_NUM : number of frames to plot
    param amp2 : amplitude 2
    param amp3 : amplitude 3
    
    Example usage:
    NO_ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_NoActivity\merged_csi_data_noactivity.csv"
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH, SUBCARRIER_NUM) # amp: signal amplitude, ts: timestamp
    plot_csi_amplitude(amp, ts, title="No Activity")
    """
    N = min(FRAME_NUM, len(amp))
    tick_spacing = 10
    ts = time_stamp[:N]

    plt.figure(figsize=(12, 6))

    if amp.ndim == 2 : 
        SUBCARRIER_NUM = amp.shape[1]
        for i in range(SUBCARRIER_NUM):
            plt.plot(ts, amp[:N, i], alpha=0.6)
            plt.xticks(ts[::tick_spacing], rotation=45)
    else : 
        valid_len = len(amp)
        plt.plot(time_stamp[1:1+valid_len], amp, alpha=0.6)
        plt.xticks( rotation=45)

    if amp2 is not None : 
        valid_len = len(amp2)
        plt.plot(time_stamp[1:1+valid_len], amp2,label="Activity Flag", linestyle='-', alpha=0.6)
    if amp3 is not None : 
        plt.axhline(amp3, color='red', linestyle='--', label=f"Threshold = {amp3:.4f}")

    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Amplitude")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%M:%S.%f"))
    plt.tight_layout()
    plt.show()
    plt.close()
    return

def plot_csi_amplitude_from_file( file_path, title="None", FRAME_NUM=500, SUBCARRIER_NUM=52):
    """Loads CSI data, calculates amplitude, and plots it.

    Example:
    ACTIVITY_CSI_PATH = r"data\Raw_CSI_To_CSV_DoorOpen\merged_csi_data_dooropen.csv"
    plot_csi_amplitude(NO_ACTIVITY_CSI_PATH, title='No Activity') # plots signal amplitude of 52 channels
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M.%S")

        I = df[[f"I{i}" for i in range(SUBCARRIER_NUM)]].values
        Q = df[[f"Q{i}" for i in range(SUBCARRIER_NUM)]].values
        amp = np.sqrt(I**2 + Q**2)

        N = min(FRAME_NUM, len(df))
        tick_spacing = 10
        ts = df["timestamp"][:N]

        plt.figure(figsize=(12, 6))
        for i in range(SUBCARRIER_NUM):
            plt.plot(ts, amp[:N, i], alpha=0.6)
        plt.title(title)
        plt.xlabel("Timestamp")
        plt.ylabel("Amplitude")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%M:%S.%f'))
        plt.xticks(ts[::tick_spacing], rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# ======realtime_csi_handler_utils 플롯 함수==========================================================
def _plot_single_feature_no_title(feat_name, feature_buffers, timestamp_buffer, 
                        topics, plot_points, feature_configs):
    """개별 특징을 플롯하는 헬퍼 함수 (제목 없이)"""
    
    for topic in topics:
        topic_label = topic.split("/")[-1]
        
        if (feat_name in feature_buffers and 
            topic in feature_buffers[feat_name] and
            topic in timestamp_buffer and
            len(timestamp_buffer[topic]) > 0 and
            len(feature_buffers[feat_name][topic]) > 0):
            
            # 데이터 가져오기
            time_data = list(timestamp_buffer[topic])[-plot_points:]
            feat_data = list(feature_buffers[feat_name][topic])[-len(time_data):]
            
            if len(feat_data) > 0:
                config = feature_configs.get(feat_name, {'label': feat_name, 'style': 'line', 'color': None})
                
                plt.plot(time_data[-len(feat_data):], feat_data,
                        label=f'{config["label"]}-{topic_label}',
                        color=config['color'], alpha=0.9, linewidth=1.5,
                        marker='o', markersize=1, markevery=10)
    
    # 개별 특징 플롯 설정 (제목 제외)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Value', fontsize=7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tick_params(axis='x', rotation=30, labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.grid(True, alpha=0.3)

def _plot_single_feature(feat_name, feature_buffers, timestamp_buffer, 
                        topics, plot_points, feature_configs):
    """개별 특징을 플롯하는 헬퍼 함수"""
    config = feature_configs[feat_name]
    
    for topic in topics:
        if (topic in feature_buffers[feat_name] and 
            len(feature_buffers[feat_name][topic]) > 0 and
            len(timestamp_buffer[topic]) > 0):
            
            feat_data = list(feature_buffers[feat_name][topic])[-plot_points:]
            time_data = list(timestamp_buffer[topic])[-len(feat_data):]
            
            if len(time_data) > 0:
                topic_label = topic.split("/")[-1]
                
                if config['style'] == 'line':
                    plt.plot(time_data, feat_data, 
                            label=f'{feat_name}-{topic_label}',
                            color=config['color'], alpha=0.9, linewidth=2,
                            marker='o', markersize=2, markevery=5)
                elif config['style'] == 'dashed':
                    plt.plot(time_data, feat_data,
                            label=f'{feat_name}-{topic_label}',
                            linestyle='--', color=config['color'], 
                            alpha=0.9, linewidth=2)
                elif config['style'] == 'step':
                    plt.step(time_data, feat_data,
                            label=f'{feat_name}-{topic_label}',
                            where='mid', color=config['color'],
                            alpha=0.7, linewidth=3)
    
    # 서브플롯 설정
    plt.title(f'{config["label"]}', fontsize=9)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel(config['label'], fontsize=7)
    plt.legend(fontsize=6)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tick_params(axis='x', rotation=30, labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.grid(True, alpha=0.3)

def _plot_cada_combined(feature_buffers, timestamp_buffer, topics, 
                       plot_points, feature_configs, cada_features):
    """CADA 특징들을 통합 플롯하는 헬퍼 함수"""
    
    for topic in topics:
        topic_label = topic.split("/")[-1]
        
        if len(timestamp_buffer[topic]) > 0:
            time_data = list(timestamp_buffer[topic])[-plot_points:]
            threshold_data = None
            
            # 각 CADA 특징 플롯
            for feat_name in cada_features:
                if (feat_name in feature_buffers and 
                    topic in feature_buffers[feat_name] and
                    len(feature_buffers[feat_name][topic]) > 0):
                    
                    feat_data = list(feature_buffers[feat_name][topic])[-len(time_data):]
                    if len(feat_data) > 0:
                        config = feature_configs[feat_name]
                        
                        if feat_name == 'activity_flag':
                            # Activity flag는 threshold 높이에 맞춰 스케일링
                            if threshold_data is not None:
                                threshold_height = np.mean(threshold_data)
                            else:
                                threshold_height = 1.0
                            feat_data = np.array(feat_data) * threshold_height
                        
                        if feat_name == 'threshold':
                            threshold_data = feat_data  # threshold 데이터 저장
                        
                        if config['style'] == 'line':
                            plt.plot(time_data[-len(feat_data):], feat_data,
                                    label=f'{config["label"]}-{topic_label}',
                                    color=config['color'], alpha=0.9, linewidth=2,
                                    marker='o', markersize=2, markevery=5)
                        elif config['style'] == 'dashed':
                            plt.plot(time_data[-len(feat_data):], feat_data,
                                    label=f'{config["label"]}-{topic_label}',
                                    linestyle='--', color=config['color'],
                                    alpha=0.9, linewidth=2)
                        elif config['style'] == 'step':
                            plt.step(time_data[-len(feat_data):], feat_data,
                                    label=f'{config["label"]}-{topic_label}',
                                    where='mid', color=config['color'],
                                    alpha=0.7, linewidth=3)
    
    # CADA 통합 플롯 설정
    plt.title('Activity Detection with Threshold', fontsize=9)
    plt.xlabel('Time', fontsize=7)
    plt.ylabel('Value', fontsize=7)
    plt.legend(fontsize=6)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.tick_params(axis='x', rotation=30, labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.grid(True, alpha=0.3)

def plot_realtime_universal(feature_buffers, timestamp_buffer, topics, 
                           plot_points=100, waiting_message=None, plot_config=None):
    """
    범용적인 실시간 플롯 함수 - 어떤 방식의 결과든 받아서 적절히 표시
    
    realtime_csi_handler_utils에서 추출된 특징들을 자동으로 감지하여 최적의 방식으로 시각화
    
    Parameters:
        feature_buffers : 특징 버퍼 딕셔너리 (realtime_csi_handler_utils에서 생성)
        timestamp_buffer : 타임스탬프 버퍼 딕셔너리
        topics : 토픽 리스트
        plot_points : 표시할 점의 개수 (기본값: 100)
        waiting_message : 대기 메시지 (None이면 정상 플롯)
        plot_config : 플롯 설정 딕셔너리 (None이면 자동 설정)
    
    Example:
        # realtime_csi_handler_utils에서 특징 추출 후
        combined_features = cada_feature_buffers  # CADA 전용 특징 딕셔너리
        plot_realtime_universal(combined_features, timestamps, topics)
    """
    
    # 대기 메시지 처리
    if waiting_message:
        plt.clf()
        plt.text(0.5, 0.5, waiting_message, ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=16)
        return
    
    # 데이터 존재 여부 확인
    has_data = any(len(timestamp_buffer[topic]) > 0 for topic in topics)
    if not has_data:
        plt.clf()
        plt.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=16)
        return
    
    plt.clf()
    
    # 자동으로 특징 타입 감지
    detected_features = []
    
    # realtime_csi_handler_utils에서 생성되는 특징들과 그 표시 방법 정의
    feature_configs = {
        # CADA 활동 감지 특징들 (초록/빨강 계열)
        'activity_detection': {'label': 'CADA Activity Detection', 'style': 'line', 'color': 'green'},
        'threshold': {'label': 'CADA Threshold', 'style': 'dashed', 'color': 'darkgreen'},
        'activity_flag': {'label': 'CADA Activity Flag', 'style': 'step', 'color': 'red'},
        
        # 미래 확장용 (다른 방법론들)
        'variance': {'label': 'Variance', 'style': 'line', 'color': None},
        'skewness': {'label': 'Skewness', 'style': 'line', 'color': None},
        'kurtosis': {'label': 'Kurtosis', 'style': 'line', 'color': None}
    }
    
    # 실제 존재하는 특징들 찾기
    for feat_name in feature_configs.keys():
        if feat_name in feature_buffers:
            # 해당 특징에 데이터가 있는지 확인
            has_feature_data = False
            for topic in topics:
                if (topic in feature_buffers[feat_name] and 
                    len(feature_buffers[feat_name][topic]) > 0):
                    has_feature_data = True
                    break
            
            if has_feature_data:
                detected_features.append(feat_name)
    
    if not detected_features:
        plt.text(0.5, 0.5, 'No valid features detected', ha='center', va='center',
                transform=plt.gca().transAxes, fontsize=16)
        return
    
    # CADA 특징들이 있으면 통합 플롯으로 처리
    cada_features = ['activity_detection', 'threshold', 'activity_flag']
    has_cada = any(feat in detected_features for feat in cada_features)
    
    if has_cada:
        # CADA 관련 세 가지 특징을 하나의 축에 통합해서 표시
        plt.subplot(1, 1, 1)
        _plot_cada_combined(
            feature_buffers,
            timestamp_buffer,
            topics,
            plot_points,
            feature_configs,
            cada_features,
        )
    else:
        # CADA 특징이 전혀 없을 경우 다른 시계열을 단순 플롯
        total_plots = len(detected_features)
        for idx, feat_name in enumerate(detected_features):
            plt.subplot(total_plots, 1, idx + 1)
            _plot_single_feature(feat_name, feature_buffers, timestamp_buffer,
                               topics, plot_points, feature_configs)
    
    plt.tight_layout(pad=0.3)


if __name__ == "__main__":
    # Example usage
    NO_ACTIVITY_CSI_PATH = (
        r"data\raw\raw_noActivity_csi\merged_csi_data_noactivity.csv"
    )
    ACTIVITY_CSI_PATH = r"data\raw\raw_activity_csi\merged_csi_data_dooropen.csv"

    # Convert and plot CSI data
    amp, ts = convert_csi_to_amplitude(NO_ACTIVITY_CSI_PATH)
    plot_csi_amplitude(amp, ts, title="No Activity")
    amp, ts = convert_csi_to_amplitude(ACTIVITY_CSI_PATH)
    plot_csi_amplitude(amp, ts, title="Door Open")
    plot_csi_amplitude_from_file(NO_ACTIVITY_CSI_PATH, title="No Activity")
    plot_csi_amplitude_from_file(ACTIVITY_CSI_PATH, title="Door Open")
