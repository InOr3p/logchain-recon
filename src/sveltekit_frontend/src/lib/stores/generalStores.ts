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

export const logs = writable<any[]>([]);
export const selectedLogs = writable<any[]>([]);
export const agents = writable<any[]>([]);