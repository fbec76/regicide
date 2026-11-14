/*
 * connection.js — reconnect-aware WebSocket wrapper.
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * string is rendered through textContent/createTextNode ONLY. innerHTML with
 * user- or server-derived data is FORBIDDEN anywhere in static/js/**.
 */
const WS_PORT = 8765;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;
const MAX_RECONNECT_ATTEMPTS = 5;

function wsUrl() {
  const host = window.location.hostname || 'localhost';
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  // The static sidecar serves HTTP on WS port + 1 (server.py), so a numeric
  // page port implies the WS port is one lower — this supports a sidecar on
  // a custom port. file:// (empty port) and other serving setups fall back
  // to the default 8765. Default serving 8766 → 8765 (unchanged).
  const port = window.location.port
    ? Number(window.location.port) - 1
    : WS_PORT;
  return `${proto}://${host}:${port}`;
}

/**
 * Create a WebSocket connection wrapper.
 *
 * @param {Object} hooks
 * @param {(data: Object) => void} hooks.onMessage  parsed JSON object (or nothing on malformed frames)
 * @param {('connected'|'disconnected') => void} hooks.onStatus
 * @returns {{ connect: () => void, send: (Object) => boolean, close: () => void }}
 */
export function createConnection({ onMessage, onStatus }) {
  let socket = null;
  let attempts = 0;
  let reconnectTimer = null;
  let userClosed = false;

  function connect() {
    window.clearTimeout(reconnectTimer);
    try {
      socket = new WebSocket(wsUrl());
    } catch {
      scheduleReconnect();
      return;
    }

    socket.addEventListener('open', () => {
      attempts = 0;
      onStatus('connected');
    });

    socket.addEventListener('message', (event) => {
      let data = null;
      try {
        data = JSON.parse(event.data);
      } catch {
        return; // malformed frame: ignore
      }
      if (data && typeof data === 'object') onMessage(data);
    });

    socket.addEventListener('close', () => {
      if (userClosed) return;
      onStatus('disconnected');
      scheduleReconnect();
    });
    // 'error' always precedes 'close'; recovery is handled there.
  }

  function scheduleReconnect() {
    if (attempts >= MAX_RECONNECT_ATTEMPTS) return; // banner offers manual recovery
    const delay = Math.min(RECONNECT_BASE_MS * 2 ** attempts, RECONNECT_MAX_MS);
    attempts += 1;
    reconnectTimer = window.setTimeout(connect, delay);
  }

  function send(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
      return true;
    }
    return false;
  }

  function close() {
    userClosed = true;
    window.clearTimeout(reconnectTimer);
    if (socket) socket.close();
  }

  return { connect, send, close };
}
