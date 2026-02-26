import boto3
import logging
import pprint
import json
import pandas as pd
import uuid
from datetime import datetime
from agent import invoke_agent_helper

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
dynamodb = boto3.resource('dynamodb')

# Replace with your values
table_name = "RestaurantBookings"
agent_id = "REPLACE_WITH_YOUR_AGENT_ID"
alias_id = "REPLACE_WITH_YOUR_AGENT_ALIAS_ID"

def selectAllFromDynamodb():
    # Get the table object
    table = dynamodb.Table(table_name)

    # Scan the table and get all items
    response = table.scan()
    items = response['Items']

    # Handle pagination if necessary
    while 'LastEvaluatedKey' in response:
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        items.extend(response['Items'])

    return pd.DataFrame(items)

# Get all bookings
items = selectAllFromDynamodb()
print(items)

# Today's date
today = datetime.today().strftime('%Y-%m-%d')
print(f"Today: {today}")

# Reserving a table for tomorrow
session_id = str(uuid.uuid1())
query = "I want to create a booking for 2 people, at 8pm tomorrow."
session_state = {"promptSessionAttributes": {"name": "John", "today": today}}

response = invoke_agent_helper(query, session_id, agent_id, alias_id, session_state=session_state)
print(response)
pprint.pprint(response)