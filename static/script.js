// script.js
// Fetches latest stats from the AI every second
// and updates the dashboard automatically

function updateDashboard() {
  fetch('/api/stats')
    .then(res => res.json())
    .then(data => {

      // ── Update stat cards ──
      document.getElementById('detection-rate').textContent =
        data.detection_rate + '%';
      document.getElementById('accuracy').textContent =
        data.accuracy + '%';
      document.getElementById('attacks-caught').textContent =
        data.attacks_caught.toLocaleString();
      document.getElementById('attacks-missed').textContent =
        data.attacks_missed.toLocaleString();
      document.getElementById('false-alarms').textContent =
        data.false_alarms.toLocaleString();
      document.getElementById('total').textContent =
        data.total.toLocaleString();

      // ── Update progress bars ──
      setBar('bar-detect', 'bar-detect-val', data.detection_rate);
      setBar('bar-acc',    'bar-acc-val',    data.accuracy);

      const faRate = data.total > 0
        ? (data.false_alarms / data.total * 100).toFixed(1)
        : 0;
      setBar('bar-fa', 'bar-fa-val', faRate);

      // ── Update live event table ──
      const tbody = document.getElementById('events-table');
      if (data.recent_events && data.recent_events.length > 0) {
        tbody.innerHTML = data.recent_events.map(e => `
          <tr>
            <td>#${e.id}</td>
            <td>${e.label === 'Attack'
              ? '🔴 Attack' : '🟢 Normal'}</td>
            <td>${e.action}</td>
            <td>${e.reward > 0
              ? '<span style="color:#2ecc71">+' + e.reward + '</span>'
              : '<span style="color:#e74c3c">'  + e.reward + '</span>'
            }</td>
            <td><span class="badge badge-${e.status}">
              ${e.status === 'success' ? '✅ Correct'
              : e.status === 'danger'  ? '❌ Missed'
              : '⚠️ False Alarm'}
            </span></td>
          </tr>
        `).join('');
      }

      // ── Update summary text ──
      const summary = document.getElementById('summary-text');
      const dr = data.detection_rate;
      if (dr >= 99) {
        summary.textContent =
          `🔥 Exceptional! AI has analyzed ${data.total.toLocaleString()} ` +
          `connections with ${dr}% detection rate. ` +
          `Only ${data.attacks_missed} attacks slipped through.`;
      } else if (dr >= 95) {
        summary.textContent =
          `✅ Excellent performance! Caught ${data.attacks_caught.toLocaleString()} ` +
          `attacks out of ${(data.attacks_caught + data.attacks_missed).toLocaleString()} total.`;
      } else {
        summary.textContent =
          `📈 AI is analyzing traffic. ` +
          `${data.total.toLocaleString()} connections processed so far.`;
      }
    });
}

function setBar(barId, valId, value) {
  document.getElementById(barId).style.width   = Math.min(value, 100) + '%';
  document.getElementById(valId).textContent   = value + '%';
}

// Update every 1 second
setInterval(updateDashboard, 1000);
updateDashboard(); // Run immediately on load