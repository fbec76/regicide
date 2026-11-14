/*
 * dom.js — tiny DOM element factory shared by all renderers.
 *
 * SECURITY RULE (plan B3.7 / B5 acceptance criterion): every server-provided
 * or user-provided string is rendered through textContent/createTextNode
 * ONLY. innerHTML with user- or server-derived data is FORBIDDEN anywhere
 * in static/js/**. This factory is text-only by construction.
 */

/**
 * Create an element with text content, attributes and children — no markup
 * parsing anywhere, so no injection path exists.
 */
export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  if (props.className !== undefined) node.className = props.className;
  if (props.text !== undefined) node.textContent = props.text;
  if (props.title !== undefined) node.title = props.title;
  if (props.disabled) node.disabled = true;
  if (props.attrs) {
    for (const [name, value] of Object.entries(props.attrs)) {
      if (value !== undefined) node.setAttribute(name, String(value));
    }
  }
  if (props.style) {
    for (const [key, value] of Object.entries(props.style)) {
      node.style[key] = value;
    }
  }
  for (const child of children) {
    if (child) node.appendChild(child);
  }
  return node;
}

/** Replace all children of a static container with the given nodes. */
export function replaceChildren(node, children) {
  while (node.firstChild) node.removeChild(node.firstChild);
  for (const child of children) {
    if (child) node.appendChild(child);
  }
}
