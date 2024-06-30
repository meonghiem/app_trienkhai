from fastapi import FastAPI
import pika.exceptions
import json

import pika.exceptions
app = FastAPI()

def send_message(msg):
    import pika
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='queue', durable=True)
    except (pika.exceptions.AMQPConnectionError) as e:
        raise Exception("Exception while connecting to Rabbit")
    except pika.exceptions.AMQPChannelError as e:
        if connection.is_closing or connection.is_closed:
            connection.close()
        raise Exception("Exception while opening channel")
    else:
        try:
            channel.confirm_delivery()
            channel.basic_publish(
                exchange='',
                routing_key='queue',
                body = json.dumps(msg),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent)
            )
        except pika.exceptions.NackError as e:
            raise Exception("Message rejected by the server")
        finally:
            connection.close()

@app.get("/")
def read_root():
    return {"Hello: World"}