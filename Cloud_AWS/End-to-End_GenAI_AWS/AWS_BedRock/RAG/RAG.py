import os
import boto3
import json
import time
import random
import string
from botocore.exceptions import ClientError
from typing import Optional

# AWS clients
bedrock_agent = boto3.client("bedrock-agent")
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")
aoss = boto3.client("opensearchserverless")
s3 = boto3.client("s3")
iam = boto3.client("iam")

STATE_FILE = "kb_state.json"


class KnowledgeBase:
    def __init__(self, name: str, embedding_model: str, region: str = "us-east-1"):
        self.name = name
        self.embedding_model = embedding_model
        self.region = region
        self.kb_id: Optional[str] = None
        self.ds_id: Optional[str] = None
        self.s3_bucket: Optional[str] = None
        self.collection_id: Optional[str] = None
        self.collection_arn: Optional[str] = None
        self.index_name: Optional[str] = None
        self.role_arn: Optional[str] = None
        self.role_name: Optional[str] = None

        self._load_state()

    # ----------------- Persistence -----------------
    def _save_state(self):
        state = {
            "kb_id": self.kb_id,
            "ds_id": self.ds_id,
            "s3_bucket": self.s3_bucket,
            "collection_id": self.collection_id,
            "collection_arn": self.collection_arn,
            "index_name": self.index_name,
            "role_arn": self.role_arn,
            "role_name": self.role_name,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE) as f:
            state = json.load(f)
        self.kb_id = state.get("kb_id")
        self.ds_id = state.get("ds_id")
        self.s3_bucket = state.get("s3_bucket")
        self.collection_id = state.get("collection_id")
        self.collection_arn = state.get("collection_arn")
        self.index_name = state.get("index_name")
        self.role_arn = state.get("role_arn")
        self.role_name = state.get("role_name")

    def _random_suffix(self, length=6):
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    # ----------------- S3 bucket -----------------
    def create_or_get_s3_bucket(self):
        if self.s3_bucket:
            try:
                s3.head_bucket(Bucket=self.s3_bucket)
                print(f"[INFO] Reusing bucket {self.s3_bucket}")
                return self.s3_bucket
            except ClientError:
                print(f"[WARN] Saved bucket {self.s3_bucket} not found, creating new.")

        bucket_name = f"{self.name.lower()}-{self._random_suffix()}"
        print(f"[INFO] Creating bucket {bucket_name} in {self.region} ...")
        if self.region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.region},
            )
        self.s3_bucket = bucket_name
        self._save_state()
        return bucket_name

    # ----------------- IAM role -----------------
    def create_or_get_role(self):
        if self.role_arn:
            try:
                iam.get_role(RoleName=self.role_name)
                print(f"[INFO] Reusing IAM role {self.role_name}")
                return self.role_arn
            except ClientError:
                print(f"[WARN] Saved IAM role {self.role_name} not found, creating new.")

        role_name = f"{self.name}-KBRole-{self._random_suffix()}"
        assume_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        print(f"[INFO] Creating IAM role {role_name} ...")
        role = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_policy),
        )
        self.role_arn = role["Role"]["Arn"]
        self.role_name = role_name

        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{self.s3_bucket}",
                        f"arn:aws:s3:::{self.s3_bucket}/*",
                    ],
                },
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": [self.collection_arn or "*"],
                },
            ],
        }
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=f"{role_name}-policy",
            PolicyDocument=json.dumps(policy_doc),
        )
        self._save_state()
        return self.role_arn

    # ----------------- AOSS collection & index -----------------
    def create_or_get_aoss_collection(self):
        if self.collection_id:
            try:
                details = aoss.batch_get_collection(ids=[self.collection_id])
                if details["collectionDetails"]:
                    print(f"[INFO] Reusing AOSS collection {self.collection_id}")
                    return self.collection_id
            except ClientError:
                print("[WARN] Saved collection not found, creating new.")

        col_name = f"{self.name}-col-{self._random_suffix()}"
        print(f"[INFO] Creating AOSS collection {col_name} ...")
        resp = aoss.create_collection(
            name=col_name, type="VECTORSEARCH", description="KB vector collection"
        )
        self.collection_id = resp["createCollectionDetail"]["id"]
        self.collection_arn = resp["createCollectionDetail"]["arn"]

        while True:
            status = aoss.batch_get_collection(ids=[self.collection_id])[
                "collectionDetails"
            ][0]["status"]
            print(f"[INFO] Collection status: {status}")
            if status == "ACTIVE":
                break
            time.sleep(10)

        self._save_state()
        return self.collection_id

    def create_or_get_index(self):
        if self.index_name:
            print(f"[INFO] Reusing index {self.index_name}")
            return self.index_name

        index_name = f"{self.name}-index-{self._random_suffix()}"
        print(f"[INFO] Creating index {index_name} ...")

        # Note: For AOSS VECTORSEARCH, index creation is via OpenSearch APIs (not boto3).
        # Here we just store logical name; Bedrock handles vector index provisioning.
        self.index_name = index_name
        self._save_state()
        return index_name

    # ----------------- Bedrock KB & Data Source -----------------
    def create_or_get_knowledge_base(self):
        if self.kb_id:
            try:
                bedrock_agent.get_knowledge_base(knowledgeBaseId=self.kb_id)
                print(f"[INFO] Reusing Knowledge Base {self.kb_id}")
                return self.kb_id
            except ClientError:
                print("[WARN] Saved KB not found, creating new.")

        print("[INFO] Creating Knowledge Base in Bedrock ...")
        resp = bedrock_agent.create_knowledge_base(
            name=self.name,
            roleArn=self.role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": f"arn:aws:bedrock:{self.region}::foundation-model/{self.embedding_model}"
                },
            },
            storageConfiguration={
                "type": "OPENSEARCHSERVERLESS",
                "opensearchServerlessConfiguration": {
                    "collectionArn": self.collection_arn,
                    "vectorIndexName": self.index_name,
                    "fieldMapping": {
                        "vectorField": "bedrock-knowledge-base-default-vector",
                        "textField": "bedrock-knowledge-base-default-text",
                        "metadataField": "bedrock-knowledge-base-default-metadata",
                    },
                },
            },
        )
        self.kb_id = resp["knowledgeBase"]["knowledgeBaseId"]
        self._save_state()
        return self.kb_id

    def create_or_get_data_source(self):
        if self.ds_id:
            try:
                bedrock_agent.get_data_source(
                    knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id
                )
                print(f"[INFO] Reusing Data Source {self.ds_id}")
                return self.ds_id
            except ClientError:
                print("[WARN] Saved Data Source not found, creating new.")

        print("[INFO] Creating Data Source ...")
        resp = bedrock_agent.create_data_source(
            knowledgeBaseId=self.kb_id,
            name=f"{self.name}-ds",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {"bucketArn": f"arn:aws:s3:::{self.s3_bucket}"},
            },
        )
        self.ds_id = resp["dataSource"]["dataSourceId"]
        self._save_state()
        return self.ds_id

    # ----------------- Sync documents -----------------
    def synchronize_data(self):
        print("[INFO] Starting sync ...")
        sync = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id
        )
        job_id = sync["ingestionJob"]["ingestionJobId"]

        while True:
            jobs = bedrock_agent.list_ingestion_jobs(knowledgeBaseId=self.kb_id)[
                "ingestionJobSummaries"
            ]
            job = next(j for j in jobs if j["ingestionJobId"] == job_id)
            print(f"[INFO] Ingestion status: {job['status']}")
            if job["status"] in ["COMPLETE", "FAILED"]:
                break
            time.sleep(10)
        return job_id

    # ----------------- Retrieval -----------------
    def rag_query(self, query: str):
        print(f"[INFO] Running RAG query: {query}")
        resp = bedrock_agent_runtime.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "knowledgeBaseConfiguration": {"knowledgeBaseId": self.kb_id},
                "type": "KNOWLEDGE_BASE",
            },
        )
        return resp.get("output", {}).get("text", "[No text output returned]")

    # ----------------- Cleanup -----------------
    def cleanup(self):
        print("[CLEANUP] Starting teardown...")

        if self.ds_id:
            try:
                bedrock_agent.delete_data_source(
                    knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id
                )
                print(f"[CLEANUP] Deleted Data Source {self.ds_id}")
            except ClientError as e:
                print(f"[CLEANUP WARN] {e}")

        if self.kb_id:
            try:
                bedrock_agent.delete_knowledge_base(knowledgeBaseId=self.kb_id)
                print(f"[CLEANUP] Deleted Knowledge Base {self.kb_id}")
            except ClientError as e:
                print(f"[CLEANUP WARN] {e}")

        if self.collection_id:
            try:
                aoss.delete_collection(id=self.collection_id)
                print(f"[CLEANUP] Deleted AOSS collection {self.collection_id}")
            except ClientError as e:
                print(f"[CLEANUP WARN] {e}")

        if self.role_name:
            try:
                iam.delete_role(RoleName=self.role_name)
                print(f"[CLEANUP] Deleted IAM Role {self.role_name}")
            except ClientError as e:
                print(f"[CLEANUP WARN] {e}")

        if self.s3_bucket:
            try:
                objs = s3.list_objects_v2(Bucket=self.s3_bucket).get("Contents", [])
                if objs:
                    s3.delete_objects(
                        Bucket=self.s3_bucket,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objs]},
                    )
                s3.delete_bucket(Bucket=self.s3_bucket)
                print(f"[CLEANUP] Deleted S3 bucket {self.s3_bucket}")
            except ClientError as e:
                print(f"[CLEANUP WARN] {e}")

        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
            print("[CLEANUP] Removed local state file.")


if __name__ == "__main__":
    kb = KnowledgeBase("DemoKB", "amazon.titan-embed-text-v1")

    # Provision resources
    kb.create_or_get_s3_bucket()
    kb.create_or_get_aoss_collection()
    kb.create_or_get_index()
    kb.create_or_get_role()
    kb.create_or_get_knowledge_base()
    kb.create_or_get_data_source()

    print(f"[OK] KB ID: {kb.kb_id}, DS ID: {kb.ds_id}")
    print(f"[INFO] Upload documents to {kb.s3_bucket} then call synchronize_data().")

    # Example usage:
    # kb.synchronize_data()
    # result = kb.rag_query("What is a kids menu?")
    # print(f"[RESULT] {result}")

    # Cleanup (uncomment to destroy resources)
    # kb.cleanup()
