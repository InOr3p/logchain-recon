// This type must match your Pydantic LogItem schema
export interface LogItem {
    id: string;
    rule_id: string;
    rule_groups: string[];
    rule_nist_800_53?: string[];
    rule_gdpr?: string[];
    rule_level: number;
    rule_firedtimes: number;
    agent_ip: string;
    data_srcip?: string;
    timestamp: string; // ISO string format
    unix_timestamp: string; // ISO string format for datetime
    rule_description: string;
}

// This matches your Pydantic Prediction schema
export interface Prediction {
    id: string;
    prediction_label: 'Benign' | 'Attack';
    prediction_score: number;
    is_attack: boolean;
}

// This matches your Pydantic ClassifyResponse schema
export interface ClassifyResponse {
    predictions: Prediction[];
}

// You can also add types for your existing functions
export interface Agent {
    id: string;
    name: string;
}

export interface AgentLogsResponse {
    count: number;
    logs: any[];
}

export interface AgentsResponse {
    agents: {
        data: {
        affected_items: Agent[];
        };
  };
}