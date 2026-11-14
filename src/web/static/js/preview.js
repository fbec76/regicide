/*
 * preview.js — live play preview (plan B3.1, Decision D2).
 *
 * The client sends {"type":"preview","card_indices":[...]} whenever the
 * selection changes during the owner's attack turn. The SERVER computes the
 * result (single source of truth for the rules) and the client renders it
 * with per-suit three-state semantics:
 *   fires    — the suit power will trigger
 *   blocked  — the power is blocked by immunity, damage still applies
 *   pending  — the spade SHIELD is held in pending_spades: spade damage
 *              applies immediately; the shield lands retroactively once a
 *              Jester negates immunity
 * Clubs have NO power icon — clubs only double damage (×2 marker).
 *
 * Server response contract (websocket_cli_bridge.compute_preview, verified
 * against tests/web/test_preview.py):
 *   valid:
 *   {"type":"preview", "valid": true, "damage": <int>,
 *    "double_damage": <bool>,
 *    "powers": {"hearts": "fires|blocked|pending", …},
 *    "defeats_enemy": <bool>, "yield_attack": <int>,
 *    "post_play_attack": <int>,        — enemy's effective attack AFTER this
 *                                        play resolves (spade shields applied
 *                                        when they fire)
 *    "pending_shield_value": <int>}    — 0 unless a spade is deferred to
 *                                        pending; that shield lands after a
 *                                        Jester
 *   invalid selection:
 *   {"type":"preview", "valid": false, "reason": "<why>"}
 *
 * SECURITY RULE (plan B3.7 / B5): every server-provided string is rendered
 * through textContent/createTextNode ONLY. innerHTML with user- or
 * server-derived data is FORBIDDEN anywhere in static/js/**.
 */
import { el, replaceChildren } from './dom.js';

const PREVIEW_DEBOUNCE_MS = 150;

// Server power states are rendered verbatim (no aliasing); unknown strings
// are dropped by parsePreview, so malformed data renders as nothing.
const SUIT_STATE_TEXT = {
  fires: 'power will fire',
  blocked: 'power blocked — damage still applies',
  pending: 'shield deferred — reduces the enemy’s attack once a Jester breaks immunity',
};

const SUIT_STATE_CLASS = {
  fires: 'pv--fires',
  blocked: 'pv--blocked',
  pending: 'pv--deferred',
};

const SUIT_LABELS = {
  hearts: '♥ Hearts',
  diamonds: '♦ Diamonds',
  spades: '♠ Spades',
};

let debounceTimer = null;

/**
 * Queue a preview request for the current selection (debounced).
 * Called after every selection change.
 */
export function requestPreview(store, connection) {
  window.clearTimeout(debounceTimer);
  if (store.state.selection.length === 0) {
    store.update({ preview: null, previewPending: false, previewError: null });
    return;
  }
  store.update({ previewPending: true, previewError: null });
  debounceTimer = window.setTimeout(() => {
    const sent = connection.send({
      type: 'preview',
      card_indices: [...store.state.selection],
    });
    if (!sent) {
      store.update({ previewPending: false, previewError: 'Not connected.' });
    }
  }, PREVIEW_DEBOUNCE_MS);
}

/** Handle a `preview` response message from the server. */
export function handlePreviewResponse(store, data) {
  store.update({
    previewPending: false,
    previewError: null,
    preview: parsePreview(data),
  });
}

/**
 * Show a preview-scoped error inline (e.g. transport failures) instead of
 * the turn log.
 */
export function showPreviewError(store, message) {
  store.update({ previewPending: false, preview: null, previewError: message });
}

/* ---- response parsing ---- */

function parsePreview(data) {
  if (!data || data.valid === false) {
    return {
      valid: false,
      reason: (data && data.reason) || 'Not a valid play.',
    };
  }
  const powers = data.powers || {};
  const suits = {};
  for (const suit of Object.keys(SUIT_LABELS)) {
    const state = powers[suit];
    if (typeof state === 'string' && SUIT_STATE_TEXT[state] !== undefined) {
      suits[suit] = state;
    }
  }
  return {
    valid: true,
    damage: typeof data.damage === 'number' ? data.damage : null,
    doubleDamage: Boolean(data.double_damage),
    suits,
    defeatsEnemy: Boolean(data.defeats_enemy),
    yieldAttack: typeof data.yield_attack === 'number' ? data.yield_attack : null,
    // post_play_attack: enemy's effective attack after this play resolves
    // (spade shields applied when they fire). pendingShieldValue: 0 unless a
    // spade is deferred to pending — that shield lands after a Jester.
    postPlayAttack: typeof data.post_play_attack === 'number'
      ? data.post_play_attack
      : null,
    pendingShieldValue: typeof data.pending_shield_value === 'number'
      ? data.pending_shield_value
      : 0,
  };
}

/* ---- rendering ---- */

/** Render the preview panel inside the play area. */
export function renderPreview(panel, state) {
  if (!state.game || state.screen !== 'table') {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const children = [el('h3', { text: 'Battle preview' })];

  if (state.previewPending) {
    children.push(el('p', { className: 'pv-note', text: 'Consulting the castle…' }));
  } else if (state.previewError) {
    children.push(
      el('p', { className: 'pv-note pv-note--muted', text: `Preview unavailable: ${state.previewError}` })
    );
  } else if (state.selection.length === 0) {
    children.push(
      el('p', { className: 'pv-note', text: 'Select cards to see what your strike will do.' })
    );
  } else if (state.preview && state.preview.valid === false) {
    children.push(
      el('p', { className: 'pv-note pv-note--muted', text: state.preview.reason })
    );
  } else if (state.preview) {
    children.push(...previewResultEls(state.preview));
  } else {
    children.push(el('p', { className: 'pv-note', text: 'Awaiting the server’s verdict…' }));
  }
  replaceChildren(panel, children);
}

function previewResultEls(preview) {
  const nodes = [];
  if (preview.damage !== null) {
    const damage = el('p', { className: 'pv-damage' }, [
      document.createTextNode('Total damage: '),
      el('strong', { text: String(preview.damage) }),
    ]);
    if (preview.doubleDamage) {
      damage.appendChild(el('span', { className: 'pv-times2', text: '×2 (clubs)' }));
    }
    nodes.push(damage);
  }
  for (const suit of Object.keys(SUIT_LABELS)) {
    const suitState = preview.suits[suit];
    if (!suitState) continue;
    nodes.push(el('p', {
      className: `pv-line ${SUIT_STATE_CLASS[suitState]}`,
      text: `${SUIT_LABELS[suit]} — ${SUIT_STATE_TEXT[suitState]}`,
    }));
  }
  if (preview.postPlayAttack !== null) {
    nodes.push(el('p', { className: 'pv-post' }, [
      document.createTextNode('Enemy attack after this play: '),
      el('strong', { text: String(preview.postPlayAttack) }),
    ]));
  }
  if (preview.pendingShieldValue > 0) {
    nodes.push(el('p', {
      className: 'pv-line pv--deferred',
      text: `Shield deferred — after a Jester: attack reduced by ${preview.pendingShieldValue} more.`,
    }));
  }
  if (preview.yieldAttack !== null) {
    nodes.push(el('p', {
      className: 'pv-line pv-yield',
      text: `If you yield: enemy strikes for ${preview.yieldAttack}.`,
    }));
  }
  if (preview.defeatsEnemy) {
    nodes.push(el('p', {
      className: 'pv-line pv--fires',
      text: 'This strike defeats the enemy!',
    }));
  }
  return nodes;
}
