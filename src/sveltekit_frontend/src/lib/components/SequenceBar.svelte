<script lang="ts">
  import { page } from '$app/stores';
  
  export let steps = [
    { name: "Classify", path: "/classify" },
    { name: "Build graph", path: "/build" },
    { name: "Predict", path: "/predict" },
    { name: "Generate report", path: "/generate" },
  ];

// ... existing script ...
  function isCompleted(index: number): boolean {
    const currentIndex = steps.findIndex(s => s.path === $page.url.pathname);
    return currentIndex !== -1 && index < currentIndex;
  }

  // Helper to determine if a step is active
  function isActive(stepPath: string): boolean {
    return $page.url.pathname === stepPath;
  }
</script>

<nav class="sequence-bar" aria-label="Project Steps">
  <ol class="steps-container">
    {#each steps as step, i}
      <!-- Step Item (Dot + Text) --><li class="step-item">
        <a 
          href={step.path}
          class="step-link"
          class:completed={isCompleted(i)}
          class:active={isActive(step.path)}
          aria-current={isActive(step.path) ? 'page' : undefined}
        >
          <div class="step-circle">
            {#if isCompleted(i)}
              <!-- Checkmark for completed steps --><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            {:else}
              <span class="step-number">{i + 1}</span>
            {/if}
          </div>
          <span class="step-name">{step.name}</span>
        </a>
      </li>
      
      <!-- Connector Line (as a sibling <li>) -->{#if i < steps.length - 1}
        <li 
          class="step-connector"
          class:completed={isCompleted(i + 1)}
        ></li>
      {/if}
    {/each}
  </ol>
</nav>

<style>
  .sequence-bar {
    padding: 0.25rem 0;
    /* Allow the container to be centered in the navbar */
    display: inline-block;
  }

  .steps-container {
    display: flex;
    /* Align to the top of the items.
      This allows us to use margin-top on the connector 
      to align it with the circle's center.
    */
    align-items: flex-start; 
    list-style: none;
    margin: 0;
    padding: 0;
    gap: 0;
    width: 100%; /* Ensure it fills its container */
  }

  /* This is the <li> holding the link */
  .step-item {
    /* No flex properties needed here */
    position: relative;
  }

  .step-link {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    text-decoration: none;
    padding: 0.25rem 0.5rem;
    transition: all 0.2s ease;
    cursor: pointer;
    position: relative;
    z-index: 2;

    /* Give all links a minimum width. 
      This ensures that links with short text ("Predict") 
      take up the same amount of space as links with 
      long text ("Generate report").
      This forces the circles to be spaced out evenly.
    */
    min-width: 90px;
  }

  .step-link:hover .step-circle {
    transform: scale(1.1);
    border-color: #3b82f6;
  }

  .step-link:hover .step-name {
    color: #ffffff;
  }

  /* Step Circle */
  .step-circle {
    /* CHANGED: Increased size */
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 2px solid #4a5568;
    background-color: #1e1e1e;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    font-weight: 600;
    font-size: 0.95rem; /* Adjusted font size for larger circle */
    color: #9ca3af;
    flex-shrink: 0;
  }

  .step-number {
    line-height: 1;
  }

  /* Completed Step */
  .step-link.completed .step-circle {
    background-color: #10b981;
    border-color: #10b981;
    color: #ffffff;
  }

  /* Active Step */
  .step-link.active .step-circle {
    background-color: #3b82f6;
    border-color: #3b82f6;
    color: #ffffff;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
  }

  /* Step Name */
  .step-name {
    font-size: 0.8rem;
    color: #9ca3af;
    font-weight: 500;
    transition: color 0.2s ease;
    white-space: nowrap;
    text-align: center;
  }

  .step-link.completed .step-name {
    color: #10b981;
  }

  .step-link.active .step-name {
    color: #ffffff;
    font-weight: 600;
  }

  /* Connector is now a sibling <li> */
  .step-connector {
    /* This makes the connector fill all available space */
    flex: 1;
    min-width: 40px; 
    
    height: 2px;
    background-color: #4a5568;
    
    /* (38px / 2) - (2px / 2) = 18px
      Increased by 2px to lower the line.
    */
    margin-top: 25px; 
    margin-left: 0.25rem;
    margin-right: 0.25rem;
    
    transition: background-color 0.3s ease;
    position: relative;
    z-index: 1;
  }

  .step-connector.completed {
    background-color: #10b981;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .step-name {
      font-size: 0.7rem;
    }

    .step-circle {
      /* CHANGED: Adjusted for smaller screens */
      width: 32px;
      height: 32px;
      font-size: 0.85rem;
    }

    .step-connector {
      min-width: 24px;
      /* (32px / 2) - (2px / 2) = 15px
        Increased by 2px to lower the line.
      */
      margin-top: 17px;
    }

    .step-link {
      padding: 0.25rem 0.25rem;
      min-width: 70px; /* Smaller min-width for mobile */
    }
  }

  @media (max-width: 576px) {
    .step-name {
      display: none;
    }

    /* CHANGED: When text is hidden, we can use align-items: center */
    .steps-container {
      align-items: center;
    }

    .step-connector {
      min-width: 20px;
      margin-top: 0; /* Let flexbox handle centering */
    }

    .step-circle {
      width: 38px;
      height: 38px;
    }
    
    .step-link {
      min-width: 0; /* No text, so no min-width needed */
      padding: 0.25rem;
      gap: 0; /* No gap needed if text is hidden */
    }
  }
</style>

