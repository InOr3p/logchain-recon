<script lang="ts">
  import { generateReport, checkReportHealth } from '$lib/controllers/graphs-controller';
  import type { AttackGraphData, AttackReport } from '$lib/schema/models';
  import { generatedReports, predictedGraphs, showAlert } from '$lib/stores/generalStores';
  import { onMount } from 'svelte';

  // State management
  let selectedGraphPath: string | null = null;
  let selectedGraphData: AttackGraphData | null = null;
  let generatedReport: AttackReport | null = null;
  let isGenerating = false;
  let error: string | null = null;
  let ollamaAvailable = false;
  let checkingHealth = true;

  // Reactive statements
  $: availableGraphs = Array.from($predictedGraphs.entries());
  $: hasGraphs = availableGraphs.length > 0;

  onMount(async () => {
    await checkOllamaHealth();
  });

  async function checkOllamaHealth() {
    checkingHealth = true;
    try {
      const health = await checkReportHealth();
      ollamaAvailable = health.ollama_available;
      if (!ollamaAvailable) {
        showAlert('Ollama service is not running. Reports cannot be generated.', 'warning', 5000);
      }
    } catch (err) {
      console.error('Error checking Ollama health:', err);
      ollamaAvailable = false;
    } finally {
      checkingHealth = false;
    }
  }

  function selectGraph(graphPath: string) {
    selectedGraphPath = graphPath;
    selectedGraphData = $predictedGraphs.get(graphPath) || null;
    generatedReport = $generatedReports.get(selectedGraphPath)!;
    error = null;
  }

  async function handleGenerateReport() {
    if (!selectedGraphData) {
      error = 'Please select a predicted graph first';
      showAlert(error, 'danger', 5000);
      return;
    }

    if (!ollamaAvailable) {
      error = 'Ollama service is not available';
      showAlert(error, 'danger', 5000);
      return;
    }

    isGenerating = true;
    error = null;
    generatedReport = null;

    try {
      const response = await generateReport(selectedGraphData);

      if (response.success && response.report) {
        generatedReport = response.report;
        $generatedReports.set(selectedGraphPath!, generatedReport)
        showAlert('Report generated successfully', 'success', 5000);
      } else {
        error = response.error || 'Failed to generate report';
        showAlert(error, 'danger', 5000);
      }
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to generate report';
      console.error('Error generating report:', err);
      showAlert(error, 'danger', 5000);
    } finally {
      isGenerating = false;
    }
  }

  function getSeverityColor(severity: string): string {
    switch (severity.toUpperCase()) {
      case 'CRITICAL': return '#dc2626';
      case 'HIGH': return '#ef4444';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': return '#10b981';
      default: return '#6b7280';
    }
  }

  function formatGraphName(path: string): string {
    const filename = path.split('/').pop()?.replace('.npz', '') || '';
    return filename.replace('inference_', '').replace(/_/g, ' ');
  }

  function exportReport() {
    if (!generatedReport) return;

    const reportText = `
ATTACK ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}
==============================================

ATTACK IDENTIFICATION
---------------------------------------------
Attack Name: ${generatedReport.attack_name}
Severity: ${generatedReport.severity}
Confidence: ${generatedReport.confidence}

SUMMARY
---------------------------------------------
${generatedReport.attack_summary}

ATTACK TIMELINE
---------------------------------------------
${generatedReport.attack_timeline.map(step => 
  `${step.step}. ${step.action}${step.timestamp ? ' (' + step.timestamp + ')' : ''}`
).join('\n')}

NIST CYBERSECURITY FRAMEWORK MAPPING
---------------------------------------------
Identify: ${generatedReport.nist_csf_mapping.Identify}
Protect: ${generatedReport.nist_csf_mapping.Protect}
Detect: ${generatedReport.nist_csf_mapping.Detect}
Respond: ${generatedReport.nist_csf_mapping.Respond}
Recover: ${generatedReport.nist_csf_mapping.Recover}

RECOMMENDED ACTIONS
---------------------------------------------
${generatedReport.recommended_actions.map((action, i) => `${i + 1}. ${action}`).join('\n')}

INDICATORS OF COMPROMISE
---------------------------------------------
${generatedReport.indicators_of_compromise.map((ioc, i) => `${i + 1}. ${ioc}`).join('\n')}
`;

    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attack_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    showAlert('Report exported successfully', 'success', 3000);
  }
</script>

<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>Attack Report Generator</h1>
    <p class=" mb-3">Generate AI-powered attack analysis reports using local Ollama</p>
    
    {#if checkingHealth}
      <div class="health-badge checking">Checking Ollama status...</div>
    {:else if ollamaAvailable}
      <div class="health-badge healthy">Ollama Available</div>
    {:else}
      <div class="health-badge unavailable">Ollama Unavailable</div>
    {/if}
  </div>

  <!-- Main Content -->
  <div class="main-content">
    <!-- Left Panel: Predicted Graphs Selection -->
    <div class="left-panel">
      <h2>Predicted Graphs ({availableGraphs.length})</h2>

      {#if !hasGraphs}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <p>No predicted graphs available</p>
          <p class="hint">Run attack prediction first in the Predict tab</p>
        </div>
      {:else}
        <div class="graph-list">
          {#each availableGraphs as [graphPath, graphData]}
            <button
              class="graph-item"
              class:selected={selectedGraphPath === graphPath}
              on:click={() => selectGraph(graphPath)}
              title={graphPath}
            >
              <div class="graph-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <line x1="16" y1="13" x2="8" y2="13"/>
                  <line x1="16" y1="17" x2="8" y2="17"/>
                  <polyline points="10 9 9 9 8 9"/>
                </svg>
              </div>
              <div class="graph-info">
                <div class="graph-name">{formatGraphName(graphPath)}</div>
                <div class="graph-stats">
                  {graphData.total_nodes} nodes • {graphData.total_edges} edges
                </div>
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Right Panel: Report Display -->
    <div class="right-panel">
      <div class="panel-header">
        <h2>Generated Report</h2>
        <div class="button-group">
          {#if generatedReport}
            <button on:click={exportReport} class="btn btn-secondary">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Export
            </button>
          {/if}
          <button
            on:click={handleGenerateReport}
            class="btn btn-primary"
            disabled={isGenerating || !selectedGraphData || !ollamaAvailable}
          >
            {isGenerating ? 'Generating...' : 'Generate Report'}
          </button>
        </div>
      </div>

      {#if !selectedGraphPath}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p>Select a predicted graph to generate a report</p>
        </div>
      {:else if isGenerating}
        <div class="loading-state">
          <div class="spinner-border"></div>
          <p>Analyzing attack sequence...</p>
          <p class="hint">This may take 30-60 seconds</p>
        </div>
      {:else if generatedReport}
        <div class="report-content">
          <!-- Header Section -->
          <div class="report-header">
            <div class="report-title-section">
              <h3 class="report-title">{generatedReport.attack_name}</h3>
              <div class="report-meta">
                <span class="severity-badge" style="background-color: {getSeverityColor(generatedReport.severity)}">
                  {generatedReport.severity}
                </span>
                <span class="confidence-badge">
                  Confidence: {generatedReport.confidence}
                </span>
              </div>
            </div>
          </div>

          <!-- Summary -->
          <div class="report-section">
            <h4>Attack Summary</h4>
            <p class="summary-text">{generatedReport.attack_summary}</p>
          </div>

          <!-- Timeline -->
          <div class="report-section">
            <h4>Attack Timeline</h4>
            <div class="timeline">
              {#each generatedReport.attack_timeline as step}
                <div class="timeline-item">
                  <div class="timeline-marker">{step.step}</div>
                  <div class="timeline-content">
                    <div class="timeline-action">{step.action}</div>
                    {#if step.timestamp}
                      <div class="timeline-timestamp">{step.timestamp}</div>
                    {/if}
                  </div>
                </div>
              {/each}
            </div>
          </div>

          <!-- NIST CSF Mapping -->
          <div class="report-section">
            <h4>NIST Cybersecurity Framework Mapping</h4>
            <div class="nist-grid">
              <div class="nist-card">
                <div class="nist-label">Identify</div>
                <div class="nist-text">{generatedReport.nist_csf_mapping.Identify}</div>
              </div>
              <div class="nist-card">
                <div class="nist-label">Protect</div>
                <div class="nist-text">{generatedReport.nist_csf_mapping.Protect}</div>
              </div>
              <div class="nist-card">
                <div class="nist-label">Detect</div>
                <div class="nist-text">{generatedReport.nist_csf_mapping.Detect}</div>
              </div>
              <div class="nist-card">
                <div class="nist-label">Respond</div>
                <div class="nist-text">{generatedReport.nist_csf_mapping.Respond}</div>
              </div>
              <div class="nist-card">
                <div class="nist-label">Recover</div>
                <div class="nist-text">{generatedReport.nist_csf_mapping.Recover}</div>
              </div>
            </div>
          </div>

          <!-- Recommended Actions -->
          <div class="report-section">
            <h4>Recommended Actions</h4>
            <ol class="actions-list">
              {#each generatedReport.recommended_actions as action}
                <li>{action}</li>
              {/each}
            </ol>
          </div>

          <!-- IOCs -->
          <div class="report-section">
            <h4>Indicators of Compromise</h4>
            <ul class="ioc-list">
              {#each generatedReport.indicators_of_compromise as ioc}
                <li class="ioc-item">{ioc}</li>
              {/each}
            </ul>
          </div>
        </div>
      {:else}
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 6v6l4 2"/>
          </svg>
          <p>Click "Generate Report" to analyze the selected attack graph</p>
        </div>
      {/if}
    </div>
  </div>
</div>

<style>
  .container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 2rem;
    font-family: "Inter", sans-serif;
  }

  .header {
    margin-bottom: 2rem;
  }

  h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0 0 0.5rem 0;
  }

  .health-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 600;
  }

  .health-badge.healthy {
    background: rgba(34, 197, 94, 0.2);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.3);
  }

  .health-badge.unavailable {
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }

  .health-badge.checking {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
  }

  .main-content {
    display: grid;
    grid-template-columns: 350px 1fr;
    gap: 1.5rem;
    min-height: 600px;
  }

  .left-panel,
  .right-panel {
    background: #1e1e1e;
    border-radius: 8px;
    border: 1px solid #333;
    overflow: hidden;
  }

  h2 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e0e0e0;
    padding: 1.25rem;
    margin: 0;
    border-bottom: 1px solid #333;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.25rem;
    border-bottom: 1px solid #333;
  }

  .panel-header h2 {
    padding: 0;
    border: none;
  }

  .button-group {
    display: flex;
    gap: 0.75rem;
  }

  .btn {
    padding: 0.625rem 1.25rem;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.95rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .btn-primary {
    background-color: #3b82f6;
    color: #fff;
  }

  .btn-primary:hover:not(:disabled) {
    background-color: #2563eb;
  }

  .btn-secondary {
    background-color: #4b5563;
    color: #fff;
  }

  .btn-secondary:hover:not(:disabled) {
    background-color: #374151;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .empty-state,
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 2rem;
    color: #666;
  }

  .empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .empty-state p {
    margin: 0.25rem 0;
    font-size: 1rem;
    color: #999;
  }

  .empty-state .hint {
    font-size: 0.875rem;
    color: #666;
  }

  .spinner-border {
    width: 40px;
    height: 40px;
    border: 3px solid #333;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .graph-list {
    max-height: 650px;
    overflow-y: auto;
  }

  .graph-item {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1.25rem;
    border: none;
    border-bottom: 1px solid #333;
    background: #1e1e1e;
    cursor: pointer;
    transition: background 0.2s;
    text-align: left;
  }

  .graph-item:hover {
    background: #2b2b2b;
  }

  .graph-item.selected {
    background: #0d0d0d;
    border-left: 3px solid #3b82f6;
  }

  .graph-icon {
    flex-shrink: 0;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #3b82f6;
    border-radius: 6px;
  }

  .graph-icon svg {
    width: 20px;
    height: 20px;
    color: white;
  }

  .graph-info {
    flex: 1;
    min-width: 0;
  }

  .graph-name {
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 0.25rem;
    text-transform: capitalize;
  }

  .graph-stats {
    font-size: 0.75rem;
    color: #666;
  }

  /* Report Content */
  .report-content {
    padding: 1.5rem;
    max-height: 700px;
    overflow-y: auto;
  }

  .report-header {
    margin-bottom: 2rem;
  }

  .report-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e0e0e0;
    margin: 0 0 1rem 0;
  }

  .report-meta {
    display: flex;
    gap: 1rem;
    align-items: center;
  }

  .severity-badge {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 700;
    color: white;
  }

  .confidence-badge {
    padding: 0.5rem 1rem;
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 6px;
    font-size: 0.875rem;
    color: #60a5fa;
  }

  .report-section {
    margin-bottom: 2rem;
  }

  .report-section h4 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #e0e0e0;
    margin: 0 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #333;
  }

  .summary-text {
    color: #d0d0d0;
    line-height: 1.6;
  }

  /* Timeline */
  .timeline {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .timeline-item {
    display: flex;
    gap: 1rem;
  }

  .timeline-marker {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    background: #3b82f6;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.875rem;
  }

  .timeline-content {
    flex: 1;
  }

  .timeline-action {
    color: #e0e0e0;
    margin-bottom: 0.25rem;
  }

  .timeline-timestamp {
    color: #666;
    font-size: 0.875rem;
  }

  /* NIST Grid */
  .nist-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .nist-card {
    background: #0d0d0d;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 1rem;
  }

  .nist-label {
    font-weight: 600;
    color: #3b82f6;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
  }

  .nist-text {
    color: #d0d0d0;
    font-size: 0.875rem;
    line-height: 1.5;
  }

  /* Actions & IOCs */
  .actions-list,
  .ioc-list {
    margin: 0;
    padding-left: 1.5rem;
  }

  .actions-list li,
  .ioc-list li {
    color: #d0d0d0;
    margin-bottom: 0.75rem;
    line-height: 1.5;
  }

  .ioc-item {
    font-family: 'Courier New', monospace;
    background: #0d0d0d;
    padding: 0.5rem;
    border-radius: 4px;
    border-left: 3px solid #ef4444;
  }
</style>