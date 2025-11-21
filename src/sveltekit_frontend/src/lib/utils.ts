import type { AttackReport } from "./schema/models";
import { showAlert } from "./stores/generalStores";


export function formatAndExportReport(generatedReport: AttackReport) {
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


export function formatDate(date: string) {
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(date));
}
