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

/**
 * Response type for the build graphs endpoint
 */
export interface BuildGraphsResponse {
  message: string;
  graph_files: string[];
}


/**
 * Graph data structure returned when fetching a specific graph
 */
export interface GraphData {
  filename: string;
  agent_ip: string;
  window_num: string;
  num_nodes: number;
  num_edges: number;
  node_feats: number[][];
  edge_index: number[][];
  log_ids: string[];
}

/**
 * Attack Graph Node - represents a node in the attack graph
 */
export interface AttackGraphNode {
  id: string;
  description: string;
  rule_groups: string[];
  rule_id: string;
  nist_800_53: string[];
  gdpr: string[];
  timestamp: string;
  max_incident_prob: number;
}

/**
 * Attack Graph Edge - represents an edge in the attack graph
 */
export interface AttackGraphEdge {
  timestamp: string;
  source_log_id: string;
  dest_log_id: string;
  edge_prob: number;
}

/**
 * Attack Graph Data - the complete attack graph response
 */
export interface AttackGraphData {
  total_nodes: number;
  total_edges: number;
  important_nodes: number;
  important_edges: number;
  sample_nodes: AttackGraphNode[];
  sample_edges: AttackGraphEdge[];
}

/**
 * Predict Attack Graph Response - response from the predict endpoint
 */
export interface PredictAttackGraphResponse {
  success: boolean;
  message: string;
  graph: AttackGraphData | null;
  threshold: number;
  metadata?: {
    graph_path: string;
    num_nodes: number;
    num_edges: number;
    num_features: number;
    num_logs_provided: number;
  };
}

