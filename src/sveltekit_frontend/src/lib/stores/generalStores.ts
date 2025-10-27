import { writable } from "svelte/store";

export let alertMessage = writable('');
export let alertColor = writable('danger');
export let alertVisible = writable(false);

export function showAlert(message = "", color = 'danger', duration = 5000) {
    alertMessage.set(message);
    alertColor.set(color);
    alertVisible.set(true);

    setTimeout(() => {
    alertVisible.set(false);
    }, duration);
}


/**
 * Stores the full list of logs fetched from the API.
 */
export const logs = writable<any[]>([]);

/**
 * Stores the list of logs currently selected by the user.
 */
export const selectedLogs = writable<any[]>([]);