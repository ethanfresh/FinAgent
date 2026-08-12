const chatEl = document.getElementById("chat");
const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const formEl = document.getElementById("composer");
const inputEl = document.getElementById("question-input");
const sendBtn = document.getElementById("send-btn");

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "msg user";
  msg.innerHTML = `<div class="bubble"></div>`;
  msg.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(msg);
  scrollToBottom();
}

function addTypingIndicator() {
  const msg = document.createElement("div");
  msg.className = "msg agent";
  msg.id = "typing-indicator";
  msg.innerHTML = `<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>`;
  messagesEl.appendChild(msg);
  scrollToBottom();
  return msg;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Renders `[title](url)` markdown links as real anchors; everything else is
// HTML-escaped plain text, so this is safe against injection from the answer text.
function renderAnswerHtml(text) {
  const escaped = escapeHtml(text);
  const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  return escaped.replace(
    linkPattern,
    (match, title, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>`
  );
}

function addAgentMessage(text, toolCalls, isError) {
  const msg = document.createElement("div");
  msg.className = "msg agent";

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (isError ? " error" : "");
  bubble.innerHTML = renderAnswerHtml(text);
  msg.appendChild(bubble);

  if (toolCalls && toolCalls.length) {
    const tags = document.createElement("div");
    tags.className = "tool-tags";
    const uniqueNames = [...new Set(toolCalls.map((t) => t.name))];
    uniqueNames.forEach((name) => {
      const tag = document.createElement("span");
      tag.className = "tool-tag";
      tag.textContent = name;
      tags.appendChild(tag);
    });
    msg.appendChild(tags);
  }

  messagesEl.appendChild(msg);
  scrollToBottom();
}

async function ask(question) {
  emptyStateEl.style.display = "none";
  addUserMessage(question);
  const typingMsg = addTypingIndicator();
  inputEl.value = "";
  inputEl.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    typingMsg.remove();

    if (!res.ok) {
      const detail = await res.text();
      addAgentMessage(`Something went wrong: ${detail || res.statusText}`, null, true);
      return;
    }

    const data = await res.json();
    addAgentMessage(data.answer, data.tool_calls, false);
  } catch (err) {
    typingMsg.remove();
    addAgentMessage(`Could not reach FinAgent: ${err.message}`, null, true);
  } finally {
    inputEl.disabled = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  ask(question);
});

document.querySelectorAll(".suggestion-chip").forEach((chip) => {
  chip.addEventListener("click", () => ask(chip.dataset.q));
});
