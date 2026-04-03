from kafka import KafkaConsumer
import json
from sqlalchemy import create_engine

# PostgreSQL connection
engine = create_engine(
    "postgresql://fraud_user:password123@localhost:5432/fraud_db"
)

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("🚀 Listening to Kafka topic...")

for message in consumer:
    data = message.value
    print("Received:", data)

    # Convert to DataFrame row
    import pandas as pd
    df = pd.DataFrame([data])

    # Insert into PostgreSQL
    df.to_sql("realtime_transactions", engine, if_exists="append", index=False)