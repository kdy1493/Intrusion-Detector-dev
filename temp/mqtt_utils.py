"""
mqtt_utils.py
------
Wi-Fi CSI MQTT 통신 유틸리티 모듈.

주요 기능
------
• start_csi_mqtt_thread 함수: MQTT 클라이언트 스레드 생성 및 구독 기능.
• publish_mock_csi 함수(있는 경우): 테스트용 모의 CSI 패킷 발행 기능.
"""

import paho.mqtt.client as mqtt
import threading


def create_mqtt_client(message_handler, topics, broker_address, broker_port):
    """범용 MQTT 클라이언트 생성 함수 (원본 util 함수 그대로)"""

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            for topic in topics:
                client.subscribe(topic)
                print(f"Subscribed to {topic}")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(client, userdata, msg):
        message_handler(msg.topic, msg.payload.decode())

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    return client


def start_mqtt_client(message_handler, topics, broker_address, broker_port):
    """MQTT 클라이언트를 메인 스레드에서 실행"""
    client = create_mqtt_client(message_handler, topics, broker_address, broker_port)
    client.connect(broker_address, broker_port, 60)
    return client


def start_mqtt_thread(message_handler, topics, broker_address, broker_port, daemon=True):
    """MQTT 루프를 백그라운드 스레드로 실행"""
    client_container = [None]

    def mqtt_thread_func():
        client = start_mqtt_client(message_handler, topics, broker_address, broker_port)
        client_container[0] = client
        client.loop_forever()

    thread = threading.Thread(target=mqtt_thread_func, daemon=daemon)
    thread.start()

    import time
    while client_container[0] is None:
        time.sleep(0.01)

    return thread, client_container[0]


# 기본 CSI 브로커 설정 -----------------------------------------------------------------
DEFAULT_CSI_BROKER = "61.252.57.136"
DEFAULT_CSI_PORT = 4991
DEFAULT_CSI_TOPICS = [
    "L0382/ESP/1",
    "L0382/ESP/2",
    "L0382/ESP/3",
    "L0382/ESP/4",
    "L0382/ESP/5",
    "L0382/ESP/6",
    "L0382/ESP/7",
    "L0382/ESP/8",
]


def start_csi_mqtt_client(message_handler, topics=None, broker_address=None, broker_port=None):
    """CSI 데이터 수신용 MQTT 클라이언트 실행 (싱글 스레드)"""
    if topics is None:
        topics = DEFAULT_CSI_TOPICS
    if broker_address is None:
        broker_address = DEFAULT_CSI_BROKER
    if broker_port is None:
        broker_port = DEFAULT_CSI_PORT

    return start_mqtt_client(message_handler, topics, broker_address, broker_port)


def start_csi_mqtt_thread(message_handler, topics=None, broker_address=None, broker_port=None, daemon=True):
    """CSI 데이터 수신용 MQTT 클라이언트를 백그라운드 스레드로 실행"""
    if topics is None:
        topics = DEFAULT_CSI_TOPICS
    if broker_address is None:
        broker_address = DEFAULT_CSI_BROKER
    if broker_port is None:
        broker_port = DEFAULT_CSI_PORT

    return start_mqtt_thread(message_handler, topics, broker_address, broker_port, daemon)
