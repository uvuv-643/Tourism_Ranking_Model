import json
import logging
import os
from time import sleep
import dotenv
import redis

from text_to_object_model import TextModel


def main():

    text_model = TextModel()
    print("Model loaded")

    dotenv.load_dotenv()
    redis_connection = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379, db=0, password=os.getenv('REDIS_PASSWORD'))
    redis_pubsub_photos = redis_connection.pubsub()
    redis_pubsub_photos.subscribe('photo_queries')
    redis_pubsub_texts = redis_connection.pubsub()
    redis_pubsub_texts.subscribe('text_queries')
    while True:
        try:
            message_photo = redis_pubsub_photos.get_message(ignore_subscribe_messages=True)
            if message_photo is not None and message_photo['data']:
                message_id = json.loads(message_photo['data'])['id']
                redis_connection.publish(f"photo_response_{message_id}", "yooou)")
        except Exception as e:
            logging.error("something went wrong", e)

        try:
            message_text = redis_pubsub_texts.get_message(ignore_subscribe_messages=True)
            if message_text is not None and message_text['data']:
                message_id = json.loads(message_text['data'])['id']
                query = json.loads(message_text['data'])['content']
                city_id = json.loads(message_text['data'])['city_id']
                predicted = text_model.predict(query, int(city_id))
                redis_connection.publish(f"text_response_{message_id}", json.dumps(predicted, ensure_ascii=False))
        except Exception as e:
            raise e

        sleep(0.1)


if __name__ == "__main__":
    main()
