/*
 * ui.js — DOM renderers for the three screens (lobby / table / game over).
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * string is rendered through textContent/createTextNode ONLY. innerHTML with
 * user- or server-derived data is FORBIDDEN anywhere in static/js/**.
 * Trusted literal markup may exist only in index.html.
 */
import {
  CASTLE_SIZE,
  canYield,
  defenseNeeded,
  handCannotCover,
  handCards,
  handSum,
  isMyTurn,
  selectedSum,
} from './state.js';
import { el, replaceChildren } from './dom.js';
import { renderPreview } from './preview.js';

const SUIT_META = {
  hearts: { symbol: '♥', color: 'red' },
  diamonds: { symbol: '♦', color: 'red' },
  clubs: { symbol: '♣', color: 'black' },
  spades: { symbol: '♠', color: 'black' },
  joker: { symbol: '✦', color: 'jester' }, // jester cards
};
const ROYAL_RANKS = new Set(['J', 'Q', 'K']);
// enemy.py: base attack is 10/15/20 for J/Q/K — exact even when shield
// exceeds it (effective_attack floors at 0).
const ROYAL_BASE_ATTACK = { J: 10, Q: 15, K: 20 };

const GAMEOVER_COPY = {
  win: 'Victory!',
  loss: 'Defeat…',
  abandoned: 'The quest is abandoned',
};
const GAMEOVER_DETAIL = {
  win: 'The castle has fallen. The realm is freed.',
  loss: 'The monarchy endures. Regroup and try again.',
  abandoned: 'A hero left the table — the quest ended for everyone.',
};

/* Animation signatures (module-level): animations must play only when the
 * underlying content changes, but renderers rebuild nodes on EVERY store
 * update (including selection/preview changes). Comparing signatures keeps
 * deal-in and new-enemy pulses one-shot instead of per-render glitches. */
let lastHandSignature = null;
let lastEnemyName = null;
let lastScreen = null;

function handSignature(cards) {
  return cards.map((card) => card.index).join(',');
}

/* ---- top-level render entry (called on every store update) ---- */

export function render(state) {
  showScreen(state);
  renderBanner(state);
  renderLobby(state);
  if (state.screen === 'table') {
    renderCastleStrip(state);
    renderEnemy(state);
    renderOpponents(state);
    renderPlayArea(state);
    renderHand(state);
    renderActionBar(state);
    renderLog(state);
  }
  renderGameOver(state);
  renderJesterModal(state);
}

function showScreen(state) {
  document.getElementById('lobby').hidden = state.screen !== 'lobby';
  document.getElementById('table').hidden = state.screen !== 'table';
  document.getElementById('gameover').hidden = state.screen !== 'gameover';
  if (state.screen === 'table' && lastScreen !== 'table') {
    // Fresh game (or re-entry): the first hand deal and enemy pulse replay.
    lastHandSignature = null;
    lastEnemyName = null;
  }
  lastScreen = state.screen;
}

/* ---- connection banner ---- */

function renderBanner(state) {
  const banner = document.getElementById('connection-banner');
  const text = document.getElementById('banner-text');
  const disconnected = state.connectionStatus === 'disconnected';
  banner.hidden = !disconnected;
  if (disconnected) {
    text.textContent = 'Connection lost — the server could not be reached.';
  }
}

/* ---- lobby ---- */

function renderLobby(state) {
  const waiting = document.getElementById('waiting');
  const forms = document.getElementById('lobby-forms');
  const hasSession = Boolean(state.session);
  waiting.hidden = !hasSession;
  forms.hidden = hasSession;
  if (!hasSession) return;

  document.getElementById('session-id').textContent = state.session.session_id;
  document.getElementById('waiting-count').textContent =
    `${state.waitingPlayers.length} / ${state.session.players_needed} heroes assembled`;
  const list = document.getElementById('waiting-players');
  replaceChildren(
    list,
    state.waitingPlayers.map((name) => el('li', { text: name }))
  );
}

/* ---- game table ---- */

function renderCastleStrip(state) {
  const strip = document.getElementById('castle-strip');
  const game = state.game;
  const enemyAlive = Boolean(game && game.enemy);
  // Shared state contract: `castle_defeated` (enemies defeated so far) and
  // `castle_total` (12, constant). Fall back to the castle_remaining
  // derivation for servers that have not shipped the new fields yet.
  const total = typeof game?.castle_total === 'number' && game.castle_total > 0
    ? game.castle_total
    : CASTLE_SIZE;
  const defeated = typeof game?.castle_defeated === 'number'
    ? game.castle_defeated
    : Math.max(CASTLE_SIZE - (game ? game.castle_remaining : CASTLE_SIZE) - (enemyAlive ? 1 : 0), 0);

  const children = [];
  for (let i = 0; i < total; i += 1) {
    let cls = 'slot--upcoming';
    if (i < defeated) cls = 'slot--defeated';
    else if (enemyAlive && i === defeated) cls = 'slot--current';
    children.push(el('span', {
      className: `castle-slot ${cls}`,
      attrs: { 'aria-hidden': 'true' },
      text: cls === 'slot--defeated' ? '✔' : '',
    }));
  }
  replaceChildren(strip, children);
  // Slots are aria-hidden decorations; the strip's own label carries the
  // progress for assistive tech.
  strip.setAttribute('aria-label', `Castle progress: ${defeated} of ${total} defeated`);
}

function renderEnemy(state) {
  const area = document.getElementById('enemy-area');
  const enemy = state.game && state.game.enemy;
  const children = [];
  if (!enemy) {
    children.push(el('p', { className: 'note', text: 'No enemy at the gates.' }));
  } else {
    children.push(enemyCardEl(enemy, state));
    children.push(statusBadgesEl(state));
    if (state.enemyHit) children.push(damagePopEl(state.enemyHit));
  }
  replaceChildren(area, children);
}

function enemyCardEl(enemy, state) {
  // Protocol exposes effective_attack and shield (no base attack). Base is
  // rank-derived for royals (enemy.py: attack 10/15/20 for J/Q/K); when
  // shield > base the bar renders as fully shielded since effective floors
  // at 0. Fallback = effective + shield, clamped for non-royals.
  const rankBase = ROYAL_BASE_ATTACK[enemy.rank];
  const baseAttack = rankBase !== undefined
    ? rankBase
    : Math.max(enemy.effective_attack + enemy.shield, enemy.effective_attack);
  const hpNow = Math.max(enemy.max_health - enemy.current_damage, 0);
  const meta = suitMeta(enemy.suit);
  const isRoyal = ROYAL_RANKS.has(enemy.rank);
  const isNew = enemy.name !== lastEnemyName;
  lastEnemyName = enemy.name;

  return el('div', {
    className: [
      'enemy-card',
      `enemy-card--${meta.color}`,
      isRoyal ? 'enemy-card--royal' : '',
      isNew ? 'enemy-card--new' : '',
      !isNew && state.enemyDefeatedFx ? 'enemy-card--defeated' : '',
    ].filter(Boolean).join(' '),
  }, [
    el('div', { className: 'enemy-card__rank', text: enemy.rank }),
    el('div', { className: 'enemy-card__suit', text: meta.symbol }),
    el('div', { className: 'enemy-card__name', text: enemy.name }),
    barEl('hp', hpNow, enemy.max_health,
      `Health ${hpNow} / ${enemy.max_health}`),
    barEl('attack', enemy.effective_attack, baseAttack,
      `Attack ${enemy.effective_attack} / ${baseAttack}`),
    enemy.shield > 0
      ? el('span', { className: 'badge badge--shield', text: `Shield ${enemy.shield}` })
      : null,
  ]);
}

/** Floating "-N" damage pop, positioned over the enemy card. */
function damagePopEl(dealt) {
  return el('span', {
    className: 'enemy-hit-pop',
    attrs: { 'aria-hidden': 'true' },
    text: `-${dealt}`,
  });
}

function statusBadgesEl(state) {
  const badges = [];
  if (state.game.immunity) {
    badges.push(el('span', { className: 'badge badge--immune', text: 'Immune — powers blocked' }));
  }
  if (state.game.face_jester_played) {
    badges.push(el('span', { className: 'badge badge--jester', text: 'Jester active' }));
  }
  return el('div', { className: 'status-badges' }, badges);
}

function barEl(kind, current, max, label) {
  const pct = max > 0 ? Math.min(Math.round((current / max) * 100), 100) : 0;
  return el('div', {
    className: `bar bar--${kind}`,
    title: label,
    attrs: { role: 'progressbar', 'aria-label': label },
    style: { },
  }, [
    el('div', { className: 'bar__fill', style: { width: `${pct}%` } }),
    // Numeric overlay (user feedback: "attack and health should include the
    // number, not only the bar") — absolutely positioned so the number can
    // never resize or shift the bar itself.
    el('span', {
      className: 'bar__num',
      text: `${current} / ${max}`,
      attrs: { 'aria-hidden': 'true' },
    }),
  ]);
}

function renderOpponents(state) {
  const left = document.getElementById('opp-left');
  const right = document.getElementById('opp-right');
  const opponents = (state.game && state.game.opponents) || [];
  const active = state.game ? state.game.active_player : null;

  const leftChildren = [];
  const rightChildren = [];
  opponents.forEach((opp, i) => {
    const isActive = opp.name === active;
    const panel = el('div', {
      className: `opp-panel${isActive ? ' opp-panel--active' : ''}`,
    }, [
      el('div', { className: 'opp-panel__name', text: opp.name }),
      el('div', { className: 'opp-panel__hand', text: `${opp.hand_size} card(s) in hand` }),
      isActive ? el('div', { className: 'opp-panel__turn', text: 'acting…' }) : null,
    ]);
    (i % 2 === 0 ? leftChildren : rightChildren).push(panel);
  });
  replaceChildren(left, leftChildren);
  replaceChildren(right, rightChildren);
}

function renderPlayArea(state) {
  const container = document.getElementById('cards-against-enemy');
  const names = (state.game && state.game.play_area) || [];
  replaceChildren(
    container,
    names.map((name) => el('span', { className: 'played-chip', text: name }))
  );
  // #cards-against-enemy is a static node — classList.toggle never restarts
  // the animation unless the flag actually flipped (cards_played).
  container.classList.toggle('play-flash', Boolean(state.playFlash));
  renderPreview(document.getElementById('preview-panel'), state);
}

function renderHand(state) {
  const handEl = document.getElementById('hand');
  const cards = handCards(state);
  const interactive = isMyTurn(state);
  // Deal-in plays once per hand change, not per re-render.
  const signature = handSignature(cards);
  const dealt = signature !== lastHandSignature;
  lastHandSignature = signature;
  replaceChildren(
    handEl,
    cards.map((card) => handCardEl(card, state, interactive, dealt))
  );
}

function handCardEl(card, state, interactive, dealt) {
  const meta = suitMeta(card.suit);
  const selected = state.selection.includes(card.index);
  const className = [
    'hand-card',
    'card',
    `card--${meta.color}`,
    card.c_type === 'jester' ? 'card--jester' : '',
    selected ? 'card--selected' : '',
    interactive ? '' : 'card--idle',
    dealt ? 'card--dealt' : '',
  ].filter(Boolean).join(' ');

  return el('button', {
    className,
    disabled: !interactive,
    title: card.display_name,
    attrs: { type: 'button', 'data-index': card.index },
  }, [
    el('span', { className: 'card__index', text: `#${card.index}` }),
    el('span', { className: 'card__rank', text: card.rank }),
    el('span', { className: 'card__suit', text: meta.symbol }),
    el('span', { className: 'card__name', text: card.display_name }),
    el('span', { className: 'card__value', text: `value ${card.value}` }),
  ]);
}

function renderActionBar(state) {
  const bar = document.getElementById('action-bar');
  if (!state.game) {
    replaceChildren(bar, []);
    bar.classList.remove('action-bar--my-turn');
    return;
  }
  const phase = state.action
    ? state.action.phase
    : state.game.phase;
  const myTurn = isMyTurn(state);
  bar.classList.toggle('action-bar--my-turn', myTurn); // static node: glow pulses once per turn change
  const children = phase === 'defense'
    ? defenseControls(state, myTurn)
    : attackControls(state, myTurn);
  replaceChildren(bar, children);
}

function attackControls(state, myTurn) {
  if (!myTurn) {
    return [el('p', {
      className: 'action-hint',
      text: `Waiting for ${state.game.active_player}…`,
    })];
  }
  const hasSelection = state.selection.length > 0;
  const yieldAllowed = canYield(state);
  return [
    el('p', { className: 'action-hint', text: 'Your turn — select cards, then strike.' }),
    el('button', {
      className: 'btn btn--primary',
      text: 'Play cards',
      disabled: !hasSelection,
      title: hasSelection ? undefined : 'Select at least one card first',
      attrs: { type: 'button', 'data-action': 'play' },
    }),
    el('button', {
      className: 'btn',
      text: 'Yield turn',
      disabled: !yieldAllowed,
      title: yieldAllowed ? undefined : 'Yielding is forbidden right now',
      attrs: { type: 'button', 'data-action': 'yield' },
    }),
  ];
}

function defenseControls(state, myTurn) {
  const needed = defenseNeeded(state) || 0;
  if (!myTurn) {
    return [el('p', {
      className: 'action-hint',
      text: `Waiting for ${state.game.active_player} to defend…`,
    })];
  }
  const sum = selectedSum(state);
  const cannotCover = handSum(state) < needed;
  // Client hardening: when the hand cannot cover the attack the loss is
  // inevitable (the server auto-terminates before asking for a defense).
  // Surface that explicitly — and keep the button enabled with a warning so
  // a stale defense action_required can still be sent and resolved by the
  // server regardless of message ordering.
  const canDefend = sum >= needed || cannotCover;
  const children = [
    el('p', {
      className: 'action-hint',
      text: `Defense! Discard cards worth at least ${needed} (selected: ${sum}).`,
    }),
  ];
  if (cannotCover) {
    children.push(el('p', {
      className: 'action-hint action-hint--danger',
      text: 'The hand cannot cover this attack — the party is lost.',
    }));
  }
  children.push(el('button', {
    className: cannotCover ? 'btn btn--primary btn--warn' : 'btn btn--primary',
    text: 'Discard to defend',
    disabled: !canDefend,
    title: cannotCover
      ? 'The hand cannot cover this attack — sending the attempt anyway'
      : (canDefend ? undefined : `Selected ${sum} of ${needed} needed`),
    attrs: { type: 'button', 'data-action': 'defend' },
  }));
  return children;
}

function renderLog(state) {
  const logEl = document.getElementById('log');
  replaceChildren(
    logEl,
    state.log.map((entry) => el('li', { className: 'log-entry' }, [
      el('span', {
        className: 'log-entry__turn',
        text: `T${entry.turn}`,
      }),
      el('span', { className: 'log-entry__text', text: entry.text }),
    ]))
  );
  logEl.scrollTop = logEl.scrollHeight; // keep the newest entries visible
}

/* ---- game over ---- */

function renderGameOver(state) {
  const result = state.ended ? state.ended.result : null;
  const section = document.getElementById('gameover');
  // Static node: flourish classes stick until the next game changes them.
  section.classList.toggle('gameover--win', result === 'win');
  section.classList.toggle('gameover--loss', result === 'loss');
  document.getElementById('gameover-title').textContent =
    GAMEOVER_COPY[result] || 'The tale ends';
  document.getElementById('gameover-detail').textContent =
    GAMEOVER_DETAIL[result] || 'The table falls silent.';
}

/* ---- jester modal ---- */

function renderJesterModal(state) {
  const backdrop = document.getElementById('jester-modal');
  const list = document.getElementById('jester-players');
  const prompt = state.jesterPrompt;
  backdrop.hidden = !prompt;
  if (!prompt) {
    replaceChildren(list, []);
    return;
  }
  replaceChildren(
    list,
    (prompt.players || []).map((player) => el('button', {
      className: 'btn btn--primary',
      text: player.name,
      attrs: { type: 'button', 'data-client-id': player.client_id },
    }))
  );
}

/* ---- small helpers ---- */

function suitMeta(suit) {
  return SUIT_META[suit] || SUIT_META.clubs;
}
