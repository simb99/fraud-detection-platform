from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC = "transactions"

while True:
    data = {
        "transaction_id": random.randint(1000, 9999),
        "amount": round(random.uniform(10, 500), 2),
        "type": random.choice(["cash_out", "payment", "transfer"]),
        "is_fraud": random.choice([0, 1])
    }

    producer.send(TOPIC, value=data)
    print("Sent:", data)

    time.sleep(2)