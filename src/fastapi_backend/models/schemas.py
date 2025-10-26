from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class LogItem(BaseModel):
    id: str
    rule_id: str
    rule_groups: List[str]
    rule_nist_800_53: Optional[List[str]] = None
    rule_gdpr: Optional[List[str]] = None
    rule_level: int
    rule_firedtimes: int
    agent_ip: str
    data_srcip: Optional[str] = None
    timestamp: str
    unix_timestamp: datetime
    rule_description: str

class ClassifiedLog(LogItem):
    label: str  # "benign" or "attack"

class LogsRequest(BaseModel):
    logs: List[LogItem]
