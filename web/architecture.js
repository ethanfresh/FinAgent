const POLL_MS = 5000;

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function pollStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const data = await res.json();

    const byStatus = data.requests_by_status || {};
    setText("stat-ok", Math.round(byStatus.ok || 0));
    setText("stat-error", Math.round(byStatus.error || 0));

    const latency = data.latency || {};
    setText(
      "stat-latency",
      latency.count ? `${latency.avg_latency_seconds.toFixed(2)}s` : "—"
    );

    const toolCalls = data.tool_calls || {};
    const totalCalls = Object.values(toolCalls).reduce((a, b) => a + b, 0);
    setText("stat-calls", Math.round(totalCalls));

    document.querySelectorAll("[data-tool]").forEach((el) => {
      const tool = el.dataset.tool;
      const count = Math.round(toolCalls[tool] || 0);
      el.textContent = `${count} call${count === 1 ? "" : "s"} since server start`;
    });
  } catch (err) {
    // Backend not reachable — leave the last known values on screen.
  }
}

pollStats();
setInterval(pollStats, POLL_MS);
