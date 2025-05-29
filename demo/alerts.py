import json
import queue
import threading

class AlertManager:
    def __init__(self):
        self.alerts_queue = queue.Queue()
        self._lock = threading.Lock()
        
    def send_alert(self, code: str, message: str):
        payload = json.dumps({'code': code, 'message': message})
        with self._lock:
            self.alerts_queue.put(payload)
            print(f'alert{code}: {message}')
        
    def get_alerts_queue(self):
        return self.alerts_queue

    def get_next_alert(self, timeout=0.1):
        try:
            return self.alerts_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class AlertCodes:
    SYSTEM_STARTED = "00"
    PERSON_DETECTED = "01"
    PERSON_LOST = "02"
    STATIONARY_BEHAVIOR = "03" 