import random
import time

def generate_log():
    return {
        "timestamp": time.strftime("%H:%M:%S"),
        "level": random.choice(["INFO", "WARN", "ERROR"]),
        "value": random.randint(10, 100)
    }

def detect_anomaly(log):
    # Simple fake rule
    return log["value"] > 90