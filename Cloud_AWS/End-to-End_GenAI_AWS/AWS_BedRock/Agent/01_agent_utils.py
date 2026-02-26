# agent_utils.py
import os
import json
import time
import zipfile
import logging
import pprint
from io import BytesIO
from typing import Optional, Dict, Any

import boto3

# ---------- clients & logging ----------
iam_client = boto3.client('iam')
sts_client = boto3.client('sts')
session = boto3.session.Session()
region = session.region_name or 'us-east-1'
account_id = sts_client.get_caller_identity()["Account"]
dynamodb_client = boto3.client('dynamodb')
dynamodb_resource = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')
bedrock_agent_client = boto3.client('bedrock-agent')
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime')

logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ---------- helpers ----------
def _safe_create_policy(policy_name: str, policy_document: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create an IAM policy if not exists, otherwise return existing policy dict.
    Returns the policy dict as returned by create_policy or get_policy.
    """
    policy_json = json.dumps(policy_document)
    try:
        policy = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=policy_json)
        logger.info("Created policy %s", policy_name)
        return policy
    except iam_client.exceptions.EntityAlreadyExistsException:
        arn = f"arn:aws:iam::{account_id}:policy/{policy_name}"
        policy = iam_client.get_policy(PolicyArn=arn)
        logger.info("Policy %s already exists, returning it", policy_name)
        return policy


def _zip_lambda_source(source_file_path: str) -> bytes:
    """
    Create a zip archive in-memory containing the source_file_path.
    The archived member will be named basename(source_file_path) inside the zip.
    """
    if not os.path.isfile(source_file_path):
        raise FileNotFoundError(f"Lambda source file not found: {source_file_path}")

    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        # Use arcname so the file inside the zip is the basename
        z.write(source_file_path, arcname=os.path.basename(source_file_path))
    return buf.getvalue()


# ---------- DynamoDB ----------
def create_dynamodb(table_name: str) -> Dict[str, Any]:
    """
    Create a DynamoDB table with booking_id as primary key (string).
    Always return a consistent dict with TableName and TableStatus.
    """
    try:
        table = dynamodb_resource.create_table(
            TableName=table_name,
            KeySchema=[{'AttributeName': 'booking_id', 'KeyType': 'HASH'}],
            AttributeDefinitions=[{'AttributeName': 'booking_id', 'AttributeType': 'S'}],
            BillingMode='PAY_PER_REQUEST'
        )
        logger.info("Creating DynamoDB table %s ...", table_name)
        table.wait_until_exists()
        logger.info("Table %s created successfully", table_name)
        return {"TableName": table_name, "TableStatus": table.table_status}
    except dynamodb_client.exceptions.ResourceInUseException:
        logger.info("Table %s already exists, returning existing table", table_name)
        table = dynamodb_resource.Table(table_name)
        return {"TableName": table_name, "TableStatus": table.table_status}


# ---------- Lambda ----------
def create_lambda(lambda_function_name: str, lambda_iam_role: Dict[str, Any], source_file_path: str = "lambda_function.py") -> Dict[str, Any]:
    """
    Create a Lambda function from the provided source file. If the function already exists,
    returns the existing function configuration.
    - lambda_iam_role: result of iam.create_role or iam.get_role (dict containing 'Role' or 'Arn').
    - source_file_path: path to the lambda source file on disk.
    Returns the lambda function configuration dict.
    """
    # Resolve role ARN
    if isinstance(lambda_iam_role, dict):
        role_arn = lambda_iam_role.get('Role', {}).get('Arn') or lambda_iam_role.get('Arn') or lambda_iam_role
    else:
        role_arn = lambda_iam_role

    zip_content = _zip_lambda_source(source_file_path)
    try:
        response = lambda_client.create_function(
            FunctionName=lambda_function_name,
            Runtime='python3.12',
            Timeout=60,
            Role=role_arn,
            Code={'ZipFile': zip_content},
            Handler=f"{os.path.splitext(os.path.basename(source_file_path))[0]}.lambda_handler"
        )
        logger.info("Created Lambda function %s", lambda_function_name)
        return response
    except lambda_client.exceptions.ResourceConflictException:
        logger.info("Lambda function %s already exists, retrieving configuration", lambda_function_name)
        response = lambda_client.get_function(FunctionName=lambda_function_name)
        return response['Configuration']


def create_lambda_role(agent_name: str, dynamodb_table_name: str) -> Dict[str, Any]:
    """
    Create a role for Lambda and attach AWSLambdaBasicExecutionRole and an inline policy
    granting access to the specified DynamoDB table. Returns the role dict.
    """
    lambda_function_role = f'{agent_name}-lambda-role'
    dynamodb_access_policy_name = f'{agent_name}-dynamodb-policy'

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    assume_role_policy_document_json = json.dumps(assume_role_policy_document)

    try:
        role = iam_client.create_role(RoleName=lambda_function_role, AssumeRolePolicyDocument=assume_role_policy_document_json)
        logger.info("Created role %s", lambda_function_role)
        # Allow time for role propagation
        time.sleep(3)
    except iam_client.exceptions.EntityAlreadyExistsException:
        role = iam_client.get_role(RoleName=lambda_function_role)
        logger.info("Role %s exists, returning existing role", lambda_function_role)

    # Attach AWSLambdaBasicExecutionRole
    try:
        iam_client.attach_role_policy(RoleName=lambda_function_role, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    except Exception as e:
        logger.warning("Could not attach AWSLambdaBasicExecutionRole: %s", e)

    # Create or get DynamoDB access policy
    dynamodb_access_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:Scan"],
                "Resource": f"arn:aws:dynamodb:{region}:{account_id}:table/{dynamodb_table_name}"
            }
        ]
    }

    policy = _safe_create_policy(dynamodb_access_policy_name, dynamodb_access_policy)
    iam_client.attach_role_policy(RoleName=lambda_function_role, PolicyArn=policy['Policy']['Arn'])
    logger.info("Attached DynamoDB access policy %s to %s", dynamodb_access_policy_name, lambda_function_role)

    return role


# ---------- Agent invocation ----------
def invoke_agent_helper(query: str, session_id: str, agent_id: str, alias_id: str,
                        enable_trace: bool = False, session_state: Optional[Dict[str, Any]] = None) -> str:
    """
    Invoke an Amazon Bedrock agent and return the agent's final text output.
    Collects all chunks instead of returning just the first one.
    """
    if session_state is None:
        session_state = {}

    logger.info("Invoking agent %s (alias %s) session %s", agent_id, alias_id, session_id)
    agent_response = bedrock_agent_runtime_client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=enable_trace,
        endSession=False,
        sessionState=session_state
    )

    if enable_trace:
        logger.debug("Agent response (raw): %s", pprint.pformat(agent_response))

    event_stream = agent_response.get('completion', [])
    output_chunks = []

    try:
        for event in event_stream:
            if 'chunk' in event:
                data = event['chunk']['bytes']
                text_piece = data.decode('utf-8')
                output_chunks.append(text_piece)
                if enable_trace:
                    logger.debug("Chunk received: %s", text_piece)
            elif 'trace' in event:
                if enable_trace:
                    logger.debug("Trace event: %s", json.dumps(event['trace'], default=str))
            else:
                logger.warning("Unexpected event in stream: %s", event)

        final_answer = "".join(output_chunks).strip()
        if not final_answer:
            raise RuntimeError("No valid text found in agent completion stream")

        logger.info("Final answer -> %s", final_answer if len(final_answer) < 500 else final_answer[:500] + "...")
        return final_answer

    except Exception as e:
        logger.exception("Error while reading agent response stream: %s", e)
        raise

# ---------- Agent Role ----------
def create_agent_role(agent_name: str, agent_foundation_model: str, kb_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create an IAM role for Amazon Bedrock Agents and attach a policy allowing model invocation.
    If kb_id provided, also include bedrock retrieve permissions.
    Returns the created or existing role dict.
    """
    agent_bedrock_allow_policy_name = f"{agent_name}-ba"
    agent_role_name = f'AmazonBedrockExecutionRoleForAgents_{agent_name}'

    statements = [
        {
            "Sid": "AmazonBedrockAgentBedrockFoundationModelPolicy",
            "Effect": "Allow",
            "Action": "bedrock:InvokeModel",
            "Resource": [f"arn:aws:bedrock:{region}::foundation-model/{agent_foundation_model}"]
        }
    ]
    if kb_id:
        statements.append({
            "Sid": "QueryKB",
            "Effect": "Allow",
            "Action": ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"],
            "Resource": [f"arn:aws:bedrock:{region}:{account_id}:knowledge-base/{kb_id}"]
        })

    bedrock_agent_bedrock_allow_policy_statement = {"Version": "2012-10-17", "Statement": statements}
    policy = _safe_create_policy(agent_bedrock_allow_policy_name, bedrock_agent_bedrock_allow_policy_statement)

    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    assume_role_policy_document_json = json.dumps(assume_role_policy_document)

    try:
        role = iam_client.create_role(RoleName=agent_role_name, AssumeRolePolicyDocument=assume_role_policy_document_json)
        logger.info("Created agent role %s", agent_role_name)
        time.sleep(3)
    except iam_client.exceptions.EntityAlreadyExistsException:
        role = iam_client.get_role(RoleName=agent_role_name)
        logger.info("Agent role %s already exists, returning existing", agent_role_name)

    iam_client.attach_role_policy(RoleName=agent_role_name, PolicyArn=policy['Policy']['Arn'])
    logger.info("Attached bedrock policy %s to role %s", agent_bedrock_allow_policy_name, agent_role_name)
    return role


# ---------- Cleanup ----------
def delete_agent_roles_and_policies(agent_name: str, kb_policy_name: Optional[str] = None):
    """
    Detach & delete the agent role and lambda role and their policies. Best effort cleanup.
    """
    agent_bedrock_allow_policy_name = f"{agent_name}-ba"
    agent_role_name = f'AmazonBedrockExecutionRoleForAgents_{agent_name}'
    dynamodb_access_policy_name = f'{agent_name}-dynamodb-policy'
    lambda_function_role = f'{agent_name}-lambda-role'

    def _try_detach(role_name, policy_arn):
        try:
            iam_client.detach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            logger.info("Detached %s from %s", policy_arn, role_name)
        except Exception as e:
            logger.warning("Could not detach %s from %s: %s", policy_arn, role_name, e)

    # detach agent policies
    for policy_name in [agent_bedrock_allow_policy_name, kb_policy_name]:
        if not policy_name:
            continue
        arn = f'arn:aws:iam::{account_id}:policy/{policy_name}'
        _try_detach(agent_role_name, arn)

    # detach lambda role policies
    for policy_name in [dynamodb_access_policy_name]:
        arn = f'arn:aws:iam::{account_id}:policy/{policy_name}'
        _try_detach(lambda_function_role, arn)

    _try_detach(lambda_function_role, 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')

    # delete roles
    for role_name in [agent_role_name, lambda_function_role]:
        try:
            iam_client.delete_role(RoleName=role_name)
            logger.info("Deleted role %s", role_name)
        except Exception as e:
            logger.warning("Could not delete role %s: %s", role_name, e)

    # delete policies
    for policy_name in [agent_bedrock_allow_policy_name, kb_policy_name, dynamodb_access_policy_name]:
        if not policy_name:
            continue
        arn = f'arn:aws:iam::{account_id}:policy/{policy_name}'
        try:
            iam_client.delete_policy(PolicyArn=arn)
            logger.info("Deleted policy %s", arn)
        except Exception as e:
            logger.warning("Could not delete policy %s: %s", arn, e)


def clean_up_resources(
        table_name: str,
        lambda_function_name: str,
        agent_action_group_response: Dict[str, Any],
        agent_functions: list,
        agent_id: str,
        kb_id: Optional[str],
        alias_id: Optional[str]
):
    """
    Clean up agent resources: disable & delete action group, disassociate KB, delete alias & agent,
    delete lambda function, delete dynamodb table. Best-effort; logs issues instead of raising.
    """
    # Unpack action group info safely
    action_group = agent_action_group_response.get('agentActionGroup', {})
    action_group_id = action_group.get('actionGroupId')
    action_group_name = action_group.get('actionGroupName')

    # 1) Disassociate KB first (if present)
    if kb_id:
        try:
            bedrock_agent_client.disassociate_agent_knowledge_base(agentId=agent_id, agentVersion='DRAFT', knowledgeBaseId=kb_id)
            logger.info("Disassociated KB %s from agent %s", kb_id, agent_id)
        except Exception as e:
            logger.warning("Could not disassociate KB %s: %s", kb_id, e)

    # 2) Disable & delete action group
    if action_group_id and action_group_name:
        try:
            bedrock_agent_client.update_agent_action_group(
                agentId=agent_id,
                agentVersion='DRAFT',
                actionGroupId=action_group_id,
                actionGroupName=action_group_name,
                actionGroupExecutor={'lambda': f"arn:aws:lambda:{region}:{account_id}:function:{lambda_function_name}"},
                functionSchema={'functions': agent_functions},
                actionGroupState='DISABLED',
            )
            logger.info("Disabled action group %s", action_group_id)
        except Exception as e:
            logger.warning("Could not disable action group %s: %s", action_group_id, e)

        try:
            bedrock_agent_client.delete_agent_action_group(agentId=agent_id, agentVersion='DRAFT', actionGroupId=action_group_id)
            logger.info("Deleted action group %s", action_group_id)
        except Exception as e:
            logger.warning("Could not delete action group %s: %s", action_group_id, e)

    # 3) Delete alias
    if alias_id:
        try:
            bedrock_agent_client.delete_agent_alias(agentAliasId=alias_id, agentId=agent_id)
            logger.info("Deleted agent alias %s", alias_id)
        except Exception as e:
            logger.warning("Could not delete agent alias %s: %s", alias_id, e)

    # 4) Delete agent
    try:
        bedrock_agent_client.delete_agent(agentId=agent_id)
        logger.info("Deleted agent %s", agent_id)
    except Exception as e:
        logger.warning("Could not delete agent %s: %s", agent_id, e)

    # 5) Delete Lambda function
    try:
        lambda_client.delete_function(FunctionName=lambda_function_name)
        logger.info("Deleted Lambda function %s", lambda_function_name)
    except Exception as e:
        logger.warning("Could not delete Lambda function %s: %s", lambda_function_name, e)

    # 6) Delete DynamoDB table
    try:
        dynamodb_client.delete_table(TableName=table_name)
        logger.info("Deleting DynamoDB table %s", table_name)
        waiter = dynamodb_client.get_waiter('table_not_exists')
        waiter.wait(TableName=table_name)
        logger.info("DynamoDB table %s deleted", table_name)
    except Exception as e:
        logger.warning("Could not delete DynamoDB table %s: %s", table_name, e)
