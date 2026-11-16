'use strict';
const menu = document.getElementById('contents');
const compact = matchMedia('(max-width:800px)');
function sizeMenu() { menu.open = !compact.matches; }
sizeMenu();
compact.addEventListener('change', sizeMenu);

const field = document.getElementById('search');
const topics = document.getElementById('topics');
const matches = document.getElementById('matches');
field.addEventListener('input', () => {
  const query = field.value.toLowerCase().trim();
  matches.hidden = !query;
  topics.hidden = Boolean(query);
  matches.replaceChildren();
  if (!query) return;
  const tokens = query.split(/\s+/);
  for (const [name, section, url] of window.docEntries) {
    const candidate = (name + ' ' + section).toLowerCase();
    if (!tokens.every(token => candidate.includes(token))) continue;
    const link = document.createElement('a');
    const context = document.createElement('small');
    link.href = url;
    link.append(name);
    context.textContent = section;
    link.append(context);
    matches.append(link);
  }
  if (!matches.children.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No matching guide or function.';
    matches.append(empty);
  }
});
