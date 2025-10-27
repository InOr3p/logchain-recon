<script lang="ts">
  import LogsTable from "$lib/components/LogTable.svelte";
</script>

<div class="card bg-dark mb-4">
    <div class="card-body">
        <h2 class="display-5 fw-bold card-title">Welcome to LogChain Recon</h2>
        <p class="fs-4">
            This application provides a step-by-step process for analyzing and
            reconstructing log data.
        </p>
        <p class="fs-5">
            To begin, please load the latest logs from a single agent, then select a step from the sequence bar above.
        </p>
        <p class="mb-0">
            Your results will appear here, and detailed logs will be shown in the
            panel below.
        </p>
    </div>
</div>

<LogsTable />




