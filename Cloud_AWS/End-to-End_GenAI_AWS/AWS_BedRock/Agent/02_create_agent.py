import time
import boto3
import logging
import ipywidgets as widgets
import uuid

from agent import create_agent_role, create_lambda_role
from agent import create_dynamodb, create_lambda, invoke_agent_helper

# -------------------------------
# Clients and logging
# -------------------------------
s3_client = boto3.client('s3')
sts_client = boto3.client('sts')
session = boto3.session.Session()
region = session.region_name
account_id = sts_client.get_caller_identity()["Account"]
bedrock_agent_client = boto3.client('bedrock-agent')
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')
lambda_client = boto3.client('lambda')

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

suffix = f"{region}-{account_id}"
agent_name = 'booking-agent'
agent_role_name = f'AmazonBedrockExecutionRoleForAgents_{agent_name}'
table_name = 'restaurant_bookings'

# -------------------------------
# DynamoDB setup
# -------------------------------
create_dynamodb(table_name)

# -------------------------------
# Foundation model selector
# -------------------------------
agent_foundation_model_selector = widgets.Dropdown(
    options=[
        ('Claude 3 Sonnet', 'anthropic.claude-3-sonnet-20240229-v1:0'),
        ('Claude 3 Haiku', 'anthropic.claude-3-haiku-20240307-v1:0')
    ],
    value='anthropic.claude-3-sonnet-20240229-v1:0',
    description='FM:',
    disabled=False,
)
agent_foundation_model = agent_foundation_model_selector.value

# -------------------------------
# IAM Roles + Lambda function
# -------------------------------
lambda_iam_role = create_lambda_role(agent_name, table_name)
lambda_function_name = f'{agent_name}-lambda'
lambda_function = create_lambda(lambda_function_name, lambda_iam_role)

agent_role = create_agent_role(agent_name, agent_foundation_model)

# -------------------------------
# Agent creation
# -------------------------------
agent_description = "Agent in charge of a restaurant's table bookings"
agent_instruction = """
You are a restaurant agent, helping clients retrieve information from their booking,
create a new booking or delete an existing booking
"""

response = bedrock_agent_client.create_agent(
    agentName=agent_name,
    agentResourceRoleArn=agent_role['Role']['Arn'],
    description=agent_description,
    idleSessionTTLInSeconds=1800,
    foundationModel=agent_foundation_model,
    instruction=agent_instruction,
)
agent_id = response['agent']['agentId']
print("Agent created with ID:", agent_id)

# -------------------------------
# Define agent functions
# -------------------------------
agent_functions = [
    {
        'name': 'get_booking_details',
        'description': 'Retrieve details of a restaurant booking',
        'parameters': {
            "booking_id": {
                "description": "The ID of the booking to retrieve",
                "required": True,
                "type": "string"
            }
        }
    },
    {
        'name': 'create_booking',
        'description': 'Create a new restaurant booking',
        'parameters': {
            "date": {"description": "YYYY-MM-DD", "required": True, "type": "string"},
            "name": {"description": "Reservation name", "required": True, "type": "string"},
            "hour": {"description": "HH:MM", "required": True, "type": "string"},
            "num_guests": {"description": "Number of guests", "required": True, "type": "integer"},
        }
    },
    {
        'name': 'delete_booking',
        'description': 'Delete an existing restaurant booking',
        'parameters': {
            "booking_id": {
                "description": "The ID of the booking to delete",
                "required": True,
                "type": "string"
            }
        }
    },
]

# -------------------------------
# Agent action group
# -------------------------------
agent_action_group_response = bedrock_agent_client.create_agent_action_group(
    agentId=agent_id,
    agentVersion='DRAFT',
    actionGroupExecutor={'lambda': lambda_function['FunctionArn']},
    actionGroupName="TableBookingsActionGroup",
    functionSchema={'functions': agent_functions},
    description="Actions for table bookings (get, create, delete)"
)

print("Action group created:", agent_action_group_response['agentActionGroup']['actionGroupId'])

# -------------------------------
# Allow Bedrock to invoke Lambda
# -------------------------------
try:
    response = lambda_client.add_permission(
        FunctionName=lambda_function_name,
        StatementId=f'allow_bedrock_{agent_id}',
        Action='lambda:InvokeFunction',
        Principal='bedrock.amazonaws.com',
        SourceArn=f"arn:aws:bedrock:{region}:{account_id}:agent/{agent_id}",
    )
    print("Lambda permission added:", response['Statement'])
except Exception as e:
    print("Permission already exists?", e)

# -------------------------------
# Prepare agent (poll until ready)
# -------------------------------
bedrock_agent_client.prepare_agent(agentId=agent_id)

print("Preparing agent...")
while True:
    status_resp = bedrock_agent_client.get_agent(agentId=agent_id)
    status = status_resp['agent']['status']
    print("Agent status:", status)
    if status == 'PREPARED':
        break
    time.sleep(10)

# -------------------------------
# Create alias
# -------------------------------
alias_response = bedrock_agent_client.create_agent_alias(
    agentId=agent_id,
    agentAliasName="TestAlias"
)
alias_id = alias_response['agentAlias']['agentAliasId']
print("Alias created:", alias_id)

# -------------------------------
# Test invocation (optional)
# -------------------------------
# session_id = str(uuid.uuid1())
# query = "Hi, I want to book a table for 2 people at 8pm on May 5, 2024."
# response = invoke_agent_helper(query, session_id, agent_id, alias_id)
# print(response)
print("Agent setup complete. You can now interact with your agent using its ID and alias.")