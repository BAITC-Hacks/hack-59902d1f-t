const $ = (id) => document.getElementById(id);
const formatter = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });
const percent = new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
let currentResult = null;
let currentFilter = 'all';

function number(value) { return value == null ? '—' : formatter.format(value); }
function pct(value) { return value == null ? '—' : `${percent.format(value)}%`; }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[char]));
}
function setText(id, text) { $(id).textContent = text; }
function setNotice(message, kind = '') {
  const element = $('notice');
  element.textContent = message;
  element.className = `notice ${kind}`.trim();
}

function fileLabel(key, defaultName) {
  const input = $(`${key}-file`);
  setText(`${key}-name`, input.files.length ? `${input.files[0].name} · загружен` : `${defaultName} · по умолчанию`);
}

async function readFile(key) {
  const input = $(`${key}-file`);
  return input.files.length ? await input.files[0].text() : undefined;
}

async function runAnalysis() {
  const seed = Number($('seed').value);
  const runs = Number($('runs').value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2000000000 || !Number.isInteger(runs) || runs < 1 || runs > 30) {
    setNotice('Укажите целый seed от 0 до 2 млрд и число прогонов от 1 до 30.', 'error');
    return;
  }
  const buttons = [$('run'), $('run-top')];
  buttons.forEach(button => button.disabled = true);
  setNotice(runs > 1 ? `Выполняется ${runs} прогонов. Это может занять около минуты…` : 'Выполняется анализ данных и запуск агента…', 'loading');
  try {
    const payload = { seed, runs };
    for (const key of ['profile', 'history', 'tariffs']) {
      const contents = await readFile(key);
      if (contents !== undefined) payload[`${key}_csv`] = contents;
    }
    const response = await fetch('/api/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось выполнить анализ.');
    currentResult = data;
    render(data);
    setNotice(`Анализ завершён. ${data.summary.positive} из ${data.summary.total} прогонов с положительным результатом.`);
  } catch (error) {
    setNotice(error.message || 'Ошибка соединения с локальным сервером.', 'error');
  } finally {
    buttons.forEach(button => button.disabled = false);
  }
}

function render(data) {
  const primary = data.primary;
  setText('metric-net', number(primary.net_gain));
  setText('metric-growth', `${primary.growth_pct >= 0 ? '+' : ''}${pct(primary.growth_pct)}`);
  setText('metric-gross', number(primary.gross_lift));
  setText('metric-cost', number(primary.cost));
  setText('metric-budget', pct(primary.cost / 100000 * 100));
  setText('metric-reach', number(primary.unique_customers));
  setText('metric-coverage', pct(primary.coverage_pct));
  const status = $('result-status');
  status.className = `result-status ${primary.status === 'PASS' ? 'pass' : 'fail'}`;
  status.innerHTML = `<span class="status-dot"></span> ${primary.status === 'PASS' ? 'Положительный результат' : 'Отрицательный результат'}`;

  const max = Math.max(Math.abs(primary.gross_lift), Math.abs(primary.cost), Math.abs(primary.net_gain), 1);
  for (const [key, value] of [['gross', primary.gross_lift], ['cost', primary.cost], ['net', primary.net_gain]]) {
    setText(`bar-${key}-value`, number(value));
    $(`bar-${key}`).style.width = `${Math.max(0, Math.min(100, Math.abs(value) / max * 100))}%`;
  }
  setText('seed-label', `SEED ${primary.seed}`);
  $('stat-pilots').innerHTML = `${number(primary.pilots)} <small>/ 20</small>`;
  $('stat-final').innerHTML = `${number(primary.final_count)} <small>/ 10</small>`;
  $('stat-contacts').innerHTML = `${number(primary.contacts)} <small>/ 15 000</small>`;
  setText('data-source', data.data.source);
  setText('data-count', `${number(data.data.customers)} абонентов · ${number(data.data.tariffs)} тарифов`);

  setText('runs-label', `${data.summary.total} ${data.summary.total === 1 ? 'ПРОГОН' : 'ПРОГОНОВ'}`);
  setText('summary-median', number(data.summary.median));
  setText('summary-min', number(data.summary.minimum));
  setText('summary-positive', `${data.summary.positive} / ${data.summary.total}`);
  renderSpark(data.runs);
  renderCampaigns();
  $('download').disabled = !primary.submission.length;
}

function renderSpark(runs) {
  const area = $('spark-area');
  const width = 720, height = 112, left = 22, right = 22, top = 16, bottom = 18;
  const values = runs.map(run => run.net_gain);
  let min = Math.min(0, ...values), max = Math.max(0, ...values);
  const spread = Math.max(1, max - min);
  min -= spread * .13; max += spread * .13;
  const x = (i) => runs.length === 1 ? width / 2 : left + i * (width - left - right) / (runs.length - 1);
  const y = (v) => top + (max - v) / (max - min) * (height - top - bottom);
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(' ');
  const zeroY = y(0);
  const circles = values.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="${runs.length === 1 ? 6 : 3.5}" fill="${v > 0 ? '#2b9563' : '#d18463'}"><title>seed ${runs[i].seed}: ${number(v)}</title></circle>`).join('');
  const areaPath = runs.length > 1 ? `<polygon points="${x(0)},${height} ${points} ${x(runs.length - 1)},${height}" fill="#74bd8d1c"/>` : '';
  const line = runs.length > 1 ? `<polyline points="${points}" fill="none" stroke="#2c9866" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>` : '';
  area.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Чистый прирост по прогонам"><line x1="0" y1="${zeroY}" x2="${width}" y2="${zeroY}" stroke="#dfe8dd" stroke-dasharray="4 5"/>${areaPath}${line}${circles}</svg>`;
}

function renderCampaigns() {
  if (!currentResult) return;
  const rows = currentResult.primary.campaigns.filter(row => currentFilter === 'all' || row.kind === currentFilter);
  setText('campaign-count', `${rows.length} ${rows.length === 1 ? 'запись' : 'записей'}`);
  const body = $('campaign-body');
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty-row">В этой категории нет записей.</td></tr>';
    return;
  }
  body.innerHTML = rows.map(row => `<tr>
    <td>${escapeHtml(row.name)}</td>
    <td><span class="type-badge ${row.kind === 'Пилот' ? 'pilot' : ''}">${escapeHtml(row.kind)}</span></td>
    <td>${escapeHtml(row.segment)}</td>
    <td>${escapeHtml(row.current_tariff)} <span style="color:#aab7ab">→</span> ${escapeHtml(row.target_tariff)}</td>
    <td><span class="channel-badge">${escapeHtml(row.channel)}</span></td>
    <td class="num">${number(row.contacts)}</td>
    <td class="num">${number(row.cost)}</td>
    <td class="num">${number(row.gross_lift)}</td>
  </tr>`).join('');
}

function downloadCSV() {
  if (!currentResult) return;
  const rows = currentResult.primary.submission;
  if (!rows.length) return;
  const columns = ['campaign_name', 'filter_arpu_segment', 'filter_data_segment', 'filter_call_segment', 'filter_current_tariff', 'target_tariff', 'channel'];
  const quote = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const csv = '\ufeff' + [columns.join(','), ...rows.map(row => columns.map(column => quote(row[column])).join(','))].join('\r\n') + '\r\n';
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `submission_seed_${currentResult.primary.seed}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

for (const [key, defaultName] of [['profile', 'customer_profile.csv'], ['history', 'change_tariff.csv'], ['tariffs', 'dict_tariff.csv']]) {
  $(`${key}-file`).addEventListener('change', () => fileLabel(key, defaultName));
}
$('run').addEventListener('click', runAnalysis);
$('run-top').addEventListener('click', runAnalysis);
$('download').addEventListener('click', downloadCSV);
document.querySelectorAll('.segmented button').forEach(button => button.addEventListener('click', () => {
  currentFilter = button.dataset.filter;
  document.querySelectorAll('.segmented button').forEach(item => item.classList.toggle('selected', item === button));
  renderCampaigns();
}));
document.querySelectorAll('.nav-link').forEach(link => link.addEventListener('click', () => {
  document.querySelectorAll('.nav-link').forEach(item => item.classList.toggle('active', item === link));
}));
runAnalysis();
