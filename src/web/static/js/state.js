/*
 * state.js — central client-side store for the Regicide web client.
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * string is rendered through textContent/createTextNode ONLY. innerHTML with
 * user- or server-derived data is FORBIDDEN anywhere in static/js/**.
 * Trusted literal markup may exist only in index.html.
 */
export const MAX_LOG_ENTRIES = 5; // turn log strip shows the last ~5 events
export const CASTLE_SIZE = 12;    // 4 Jacks, 4 Queens, 4 Kings

function initialState() {
  return {
    myClientId: null,
    connectionStatus: 'connecting', // 'connecting' | 'connected' | 'disconnected'
    screen: 'lobby',                // 'lobby' | 'table' | 'gameover'
    session: null,                  // {session_id, players_needed} while waiting
    waitingPlayers: [],             // names assembled in the waiting room
    game: null,                     // last personalized `state` message
    action: null,                   // last `action_required` message
    selection: [],                  // ordered hand indices, in click order
    jesterPrompt: null,             // open jester_choice_required message
    preview: null,                  // parsed preview response (see preview.js)
    previewPending: false,
    previewError: null,
    ended: null,                    // {result: 'win'|'loss'|'abandoned'}
    log: [],                        // entries: {turn: <int>, text: <string>}
    turnNumber: 0,                  // client-side count of received `state` messages
    playFlash: false,               // transient: play-area pulse after cards_played
    enemyDefeatedFx: false,         // transient: enemy shake after enemy_defeated
    enemyHit: null,                 // transient: damage dealt to the enemy (int)
  };
}

export function createStore() {
  const state = initialState();
  const listeners = new Set();
  return {
    state,
    subscribe(fn) {
      listeners.add(fn);
    },
    update(patch) {
      Object.assign(state, patch);
      for (const fn of listeners) fn(state);
    },
  };
}

/* ---- derived values (pure functions over store state) ---- */

export function handCards(state) {
  return (state.game && state.game.you && state.game.you.hand) || [];
}

/** True when this client owns the current action (attack or defense). */
export function isMyTurn(state) {
  return Boolean(
    state.game && state.game.active_player_client_id === state.myClientId
  );
}

export function selectedSum(state) {
  const hand = handCards(state);
  return state.selection.reduce(
    (sum, i) => sum + ((hand[i] && hand[i].value) || 0),
    0
  );
}

/** Total value of the entire hand (used for the "party is lost" check). */
export function handSum(state) {
  return handCards(state).reduce((sum, card) => sum + (card.value || 0), 0);
}

/**
 * True during a defense phase when the whole hand cannot cover the attack:
 * the loss is inevitable, but the client must still be able to send the
 * defense attempt so the server can resolve it.
 */
export function handCannotCover(state) {
  const needed = defenseNeeded(state);
  return needed !== null && handSum(state) < needed;
}

export function defenseNeeded(state) {
  return state.action && state.action.phase === 'defense'
    ? state.action.needed
    : null;
}

export function canYield(state) {
  return Boolean(
    state.action &&
    state.action.phase === 'attack' &&
    state.action.can_yield === true
  );
}

/* ---- store mutations ---- */

export function pushLog(store, text) {
  const entry = { turn: store.state.turnNumber, text };
  const log = store.state.log.concat([entry]).slice(-MAX_LOG_ENTRIES);
  store.update({ log });
}

export function toggleSelection(store, index) {
  const selection = store.state.selection;
  const next = selection.includes(index)
    ? selection.filter((i) => i !== index)
    : selection.concat([index]);
  store.update({ selection: next });
}

export function clearSelection(store) {
  store.update({
    selection: [],
    preview: null,
    previewPending: false,
    previewError: null,
  });
}

/** Reset everything game-related; used for leave/disconnect recovery. */
export function resetToLobby(store) {
  store.update({
    screen: 'lobby',
    session: null,
    waitingPlayers: [],
    game: null,
    action: null,
    ended: null,
    jesterPrompt: null,
    log: [],
    selection: [],
    preview: null,
    previewPending: false,
    previewError: null,
    turnNumber: 0,
    playFlash: false,
    enemyDefeatedFx: false,
    enemyHit: null,
  });
}
