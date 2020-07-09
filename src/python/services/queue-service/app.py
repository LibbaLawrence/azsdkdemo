import requests
import os
import time
import json

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    EnvironmentCredential,
)
from azure.storage.queue import QueueServiceClient
from azure.table import TableServiceClient
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.textanalytics import TextAnalyticsClient

from smart_getenv import getenv
from dotmap import DotMap
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

sleep = getenv("AZURE_STORAGE_QUEUE_RECEIVE_SLEEP", type=bool, default=1)

credential = ChainedTokenCredential(
    AzureCliCredential(), EnvironmentCredential(), ManagedIdentityCredential()
)

queue_service_client = QueueServiceClient(
    account_url=getenv("AZURE_STORAGE_QUEUE_ENDPOINT"), credential=credential
)

queue_client = queue_service_client.get_queue_client(
    queue=getenv("AZURE_STORAGE_QUEUE_NAME", default="messages")
)

fr_client = FormRecognizerClient(
    endpoint=getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"), credential=credential
)

ta_client = TextAnalyticsClient(
    endpoint=getenv("AZURE_TEXT_ANALYTICS_ENDPOINT"), credential=credential
)

table_service_client = TableServiceClient(
    account_url=getenv("AZURE_TABLE_ENDPOINT"), credential=credential
)

table_client = table_service_client.get_table_client(table="azimageai")


while True:

    print("Receiving messages...")
    batches = queue_client.receive_messages(
        messages_per_page=getenv("AZURE_STORAGE_QUEUE_MSG_COUNT", default="10")
    )
    for batch in batches.by_page():
        for message in batch:
            message_json = DotMap(json.loads(message.content))

            fr_poller = fr_client.begin_recognize_content_from_url(message_json.url)
            fr_result = fr_poller.result()

            lines_of_text = []
            for page in fr_result:
                for line in page.lines:
                    lines_of_text.append(line.text)
            text = " ".join(lines_of_text)
            lines_of_text.clear()

            print(text)
            message_json.text = text

            if text:
                ta_response = ta_client.analyze_sentiment([text])
                for doc in ta_response:
                    print(doc.sentiment)
                    message_json.sentiment = doc.sentiment

                table_client.upsert_entity(mode="replace", table_entity_properties=message_json.toDict())

                queue_client.delete_message(message)

            print(message_json)

    time.sleep(sleep)

