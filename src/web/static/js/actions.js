/*
 * actions.js — input handling (plan B3): click-to-select, play/yield,
 * defense discard, jester chooser, leave/return-to-lobby.
 *
 * Listeners use event delegation on the static containers from index.html
 * (renderers replace container children on every store update).
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * string is rendered through textContent/createTextNode ONLY. innerHTML with
 * user- or server-derived data is FORBIDDEN anywhere in static/js/**.
 */
import {
  clearSelection,
  handCannotCover,
  isMyTurn,
  resetToLobby,
  toggleSelection,
} from './state.js';
import { requestPreview } from './preview.js';

export function initActions(store, connection) {
  wireLobbyForms(connection);
  wireHandClicks(store, connection);
  wireActionBar(store, connection);
  wireJesterModal(store, connection);
  wireChromeButtons(store, connection);
}

function wireLobbyForms(connection) {
  document.getElementById('create-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const name = document.getElementById('create-name').value.trim();
    const count = Number(document.getElementById('create-count').value);
    if (!name || ![2, 3, 4].includes(count)) return; // client-side guard only; server re-validates
    connection.send({ type: 'create_game', player_name: name, num_players: count });
  });

  document.getElementById('join-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const sessionId = document.getElementById('join-session').value.trim();
    const name = document.getElementById('join-name').value.trim();
    if (!sessionId || !name) return;
    connection.send({ type: 'join_game', session_id: sessionId, player_name: name });
  });
}

function wireHandClicks(store, connection) {
  document.getElementById('hand').addEventListener('click', (event) => {
    const cardBtn = event.target.closest('.hand-card');
    if (!cardBtn || cardBtn.disabled) return;
    if (store.state.screen !== 'table' || !isMyTurn(store.state)) return;

    const index = Number(cardBtn.getAttribute('data-index'));
    if (!Number.isInteger(index)) return;
    toggleSelection(store, index);
    requestPreview(store, connection);
  });
}

function wireActionBar(store, connection) {
  document.getElementById('action-bar').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-action]');
    if (!btn || btn.disabled) return;
    const action = btn.getAttribute('data-action');
    const { selection } = store.state;

    if (action === 'play' && selection.length > 0) {
      connection.send({ type: 'play', card_indices: [...selection] });
      clearSelection(store);
    } else if (action === 'yield') {
      // Yield = play with an empty card list (bridge maps [] → yield).
      connection.send({ type: 'play', card_indices: [] });
      clearSelection(store);
    } else if (action === 'defend'
      && (selection.length > 0 || handCannotCover(store.state))) {
      // Normally a selection is required; when the hand cannot cover the
      // attack (stale defense request), the attempt is sent even with an
      // empty selection so the server can resolve the loss.
      connection.send({ type: 'defend', card_indices: [...selection] });
      clearSelection(store);
    }
  });
}

function wireJesterModal(store, connection) {
  document.getElementById('jester-players').addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-client-id]');
    if (!btn) return;
    const targetClientId = btn.getAttribute('data-client-id');
    store.update({ jesterPrompt: null });
    connection.send({ type: 'jester_choose', target_client_id: targetClientId });
  });
}

function wireChromeButtons(store, connection) {
  const returnToLobby = () => {
    connection.send({ type: 'leave_game' }); // harmless when already out of a session
    resetToLobby(store);
  };
  document.getElementById('banner-lobby-btn').addEventListener('click', returnToLobby);
  document.getElementById('gameover-lobby-btn').addEventListener('click', returnToLobby);
  document.getElementById('leave-btn').addEventListener('click', returnToLobby);
}
