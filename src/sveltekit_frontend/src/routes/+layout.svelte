<script lang="ts">
  import Alert from "$lib/components/Alert.svelte";
  import Navbar from "$lib/components/Navbar.svelte"; 
  import "../app.css";
  import "bootstrap/dist/css/bootstrap.min.css";  

</script>

<div class="text-light vh-100 d-flex flex-column">
  
  <Navbar />
  <Alert />


  <main class="flex-grow-1 overflow-auto p-3">
    <slot />
  </main>

  <!-- Footer -->
  <footer class="bg-black text-secondary text-center py-3 flex-shrink-0">
    <small>{new Date().getFullYear()} LogChain Recon</small>
  </footer>
</div>