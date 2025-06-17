"""mqtt_utils – 초경량 MQTT 래퍼

────────────────────────────────────────────
● 역할
    ▸ paho-mqtt 클라이언트 초기화·연결·구독·수신을
      start_mqtt() 하나로 단순화.

● 공개 함수
    start_mqtt(topics, broker, port, on_packet) → (client, thread)
        ‣ 내부에서 daemon 스레드(loop_forever) 실행
        ‣ 수신 payload 는 on_packet(topic, payload) 콜백으로 그대로 전달

● 특징
    ▸ on_packet 쪽 예외를 try/except 로 감싸 로그만 출력
    ▸ 반환된 client 객체로 필요 시 disconnect(), loop_stop() 가능
────────────────────────────────────────────
"""

import paho.mqtt.client as mqtt
import threading

def start_mqtt(topics, broker, port, on_packet):
    """
    topics    : ['topic/1', 'topic/2', ...]
    on_packet : callback(topic:str, payload:str)
    """
    def _on_connect(cli, *_):
        for t in topics:
            cli.subscribe(t)

    def _on_message(cli, _u, msg):
        try:
            on_packet(msg.topic, msg.payload.decode())
        except Exception as e:
            print("MQTT cb error:", e)

    cli = mqtt.Client()
    cli.on_connect, cli.on_message = _on_connect, _on_message
    cli.connect(broker, port, 60)

    th = threading.Thread(target=cli.loop_forever, daemon=True)
    th.start()
    return cli, th
