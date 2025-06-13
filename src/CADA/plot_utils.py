"""
plot_utils.py
------
Wi-Fi CSI 데이터 시각화 모듈

기능
------
• 실시간 CSI 데이터 시각화
• 여러 AP의 데이터 동시 표시
• Activity Detection, Threshold, Activity Flag 통합 표시
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def plot_realtime_universal(feature_buffers, timestamp_buffer, topics, 
                           plot_points=100, waiting_message=None, plot_config=None):
    """
    실시간 CADA 특징 시각화 함수
    
    Parameters
    ----------
    feature_buffers : dict
        특징별 데이터 버퍼
        {
            'activity_detection': {'CSI/AP1': deque([...])},
            'threshold': {'CSI/AP1': deque([...])},
            'activity_flag': {'CSI/AP1': deque([...])}
        }
    timestamp_buffer : dict
        토픽별 타임스탬프 버퍼
        {'CSI/AP1': deque([...])}
    topics : list
        시각화할 MQTT 토픽 목록
        ['CSI/AP1', 'CSI/AP2']
    plot_points : int
        표시할 데이터 포인트 수
    waiting_message : str
        대기 중 표시 메시지
    plot_config : dict
        플롯 설정
    
    특징
    -----
    • Activity Detection: 파란색 실선
    • Threshold: 진한 녹색 점선
    • Activity Flag: 빨간색 계단형
    • Activity Flag는 Threshold 높이에 맞춰 스케일링
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
    
    # CADA 특징 설정
    feature_configs = {
        'activity_detection': {'label': 'Activity Detection', 'style': 'line', 'color': 'blue'},
        'threshold': {'label': 'Threshold', 'style': 'dashed', 'color': 'darkgreen'},
        'activity_flag': {'label': 'Activity Flag', 'style': 'step', 'color': 'red'}
    }
    
    # CADA 특징들
    cada_features = ['activity_detection', 'threshold', 'activity_flag']
    
    # CADA 특징 플롯
    plt.subplot(1, 1, 1)
    
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
                            threshold_data = feat_data
                        
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
    
    plt.tight_layout(pad=0.3)
