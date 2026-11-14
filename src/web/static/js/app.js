/*
 * app.js — bootstrap: store + connection wiring and message routing.
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * string is rendered through textContent/createTextNode ONLY. innerHTML with
 * user- or server-derived data is FORBIDDEN anywhere in static/js/**.
 * Unknown message types are tolerated silently (forward compatibility).
 */
import {
  clearSelection,
  createStore,
  pushLog,
  resetToLobby,
} from './state.js';
import { createConnection } from './connection.js';
import { render } from './ui.js';
import { initActions } from './actions.js';
import { handlePreviewResponse, showPreviewError } from './preview.js';

const store = createStore();

// Transient FX lifetimes (ms) — flags clear via timeout so the next render
// drops the animation class.
const PLAY_FLASH_MS = 600;
const DAMAGE_POP_MS = 800;

// Routing table for server → client messages (frozen Phase A protocol).
const handlers = {
  welcome(data) {
    const reconnect = store.state.connectionStatus === 'disconnected'
      || store.state.screen !== 'lobby';
    const patch = { myClientId: data.client_id, connectionStatus: 'connected' };
    if (reconnect) {
      // After a reconnect the old session is gone server-side: back to lobby.
      resetToLobby(store);
    }
    store.update(patch);
  },

  error(data) {
    const message = data.message || 'unknown error';
    // An error while the jester modal is open (e.g. the server's 60s jester
    // choice timeout) must not soft-lock it: clear the prompt and surface
    // the reason inline in the turn log.
    if (store.state.jesterPrompt) {
      store.update({ jesterPrompt: null });
      pushLog(store, `✖ ${message}`);
      return;
    }
    // While a preview is pending, a server error is almost certainly the
    // preview reply (e.g. handler not yet deployed) — show it inline.
    if (store.state.previewPending) {
      showPreviewError(store, message);
      return;
    }
    pushLog(store, `✖ ${message}`);
  },

  game_created(data) {
    store.update({
      session: {
        session_id: data.session_id,
        players_needed: data.players_needed,
      },
      waitingPlayers: [data.player_name],
    });
  },

  player_joined(data) {
    if (store.state.session) {
      store.update({
        waitingPlayers: store.state.waitingPlayers.concat([data.player_name]),
      });
    }
  },

  player_left(data) {
    if (store.state.session) {
      store.update({
        waitingPlayers: store.state.waitingPlayers.filter(
          (name) => name !== data.player_name
        ),
      });
    } else if (store.state.game) {
      pushLog(store, `${data.player_name} left the game.`);
    }
  },

  game_started() {
    store.update({
      screen: 'table',
      session: null,
      log: [],
      turnNumber: 1,
      playFlash: false,
      enemyDefeatedFx: false,
      enemyHit: null,
    });
    pushLog(store, 'The battle begins!');
  },

  state(data) {
    const patch = {
      game: data,
      action: null,
      turnNumber: store.state.turnNumber + 1,
      enemyDefeatedFx: false, // a fresh state means a fresh board
    };
    const prev = store.state.game;
    if (prev && prev.enemy && data.enemy) {
      // Damage dealt this strike = HP drop between consecutive snapshots.
      // (cards_played carries no damage field; this stays server-derived.)
      const dealt = data.enemy.current_damage - prev.enemy.current_damage;
      if (dealt > 0) {
        patch.enemyHit = dealt;
        window.setTimeout(() => store.update({ enemyHit: null }), DAMAGE_POP_MS);
      }
    }
    store.update(patch);
    clearSelection(store);
  },

  action_required(data) {
    store.update({ action: data });
  },

  jester_choice_required(data) {
    store.update({ jesterPrompt: data });
  },

  cards_played(data) {
    const cards = (data.cards || []).join(', ');
    const suffix = data.jester ? ' — the Jester dances!' : '';
    pushLog(store, `${data.player} played ${cards || 'nothing'}${suffix}`);
    store.update({ playFlash: true });
    window.setTimeout(() => store.update({ playFlash: false }), PLAY_FLASH_MS);
  },

  player_yielded(data) {
    pushLog(store, `${data.player} yielded the turn.`);
  },

  enemy_defeated(data) {
    pushLog(store, `${data.player} struck down the enemy!`);
    store.update({ enemyDefeatedFx: true }); // cleared by the next `state`
  },

  defense_resolved(data) {
    const discarded = (data.discarded || []).join(', ');
    pushLog(store, `${data.player} defended — discarded ${discarded || 'nothing'}.`);
  },

  action_rejected(data) {
    pushLog(store, `✖ ${data.error || 'action rejected'}`);
  },

  game_ended(data) {
    // Clear any open jester prompt — an abandoned game leaves no jester
    // choice to make, and the modal must not overlay the game-over screen.
    store.update({
      ended: { result: data.result },
      screen: 'gameover',
      jesterPrompt: null,
    });
  },

  preview(data) {
    handlePreviewResponse(store, data);
  },
};

const connection = createConnection({
  onMessage(data) {
    const handler = handlers[data.type];
    if (handler) handler(data); // unknown types: silently tolerated
  },
  onStatus(status) {
    store.update({ connectionStatus: status });
  },
});

store.subscribe(() => render(store.state));
initActions(store, connection);
connection.connect();
