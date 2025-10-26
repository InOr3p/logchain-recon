<script lang="ts">
  import Navbar from "$lib/components/Navbar.svelte";
import "../app.css";
  import "bootstrap/dist/css/bootstrap.min.css";

  // Load Bootstrap JS safely on the client
  /*onMount(async () => {
    await import("bootstrap/dist/js/bootstrap.bundle.min.js");
  });*/
</script>

<!-- Global App Layout -->
<div class="bg-dark text-light min-vh-100 d-flex flex-column">
  
	<Navbar></Navbar>

  <!-- Page Content -->
  <main class="flex-grow-1 container py-5">
    <slot />
  </main>

  <!-- Footer -->
  <footer class="bg-black text-secondary text-center py-3 mt-auto">
    <small>{new Date().getFullYear()} LogChain Recon</small>
  </footer>
</div>
