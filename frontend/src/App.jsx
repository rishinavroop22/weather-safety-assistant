import { useEffect, useRef, useState } from "react";
import { MAX_MESSAGE_CHARS, sendChat } from "./api.js";
import Message from "./components/Message.jsx";

const EXAMPLES = [
  "Is it safe to cycle in Mysuru today?",
  "Should I take my bike out? It's really windy.",
  "Can I take my child to the park this evening?",
];
const SESSION_KEY = "weather-safety-session-id";

function readSession() {
  try {
    return sessionStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function writeSession(id) {
  try {
    if (id) sessionStorage.setItem(SESSION_KEY, id);
    else sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // storage unavailable (private mode): the session simply lasts for this page
  }
}

let nextId = 0;
const newId = () => `m${++nextId}`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState(readSession);
  const endRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading]);

  async function send(text) {
    const message = text.trim();
    if (!message || loading) return;
    if (message.length > MAX_MESSAGE_CHARS) return;

    setMessages((current) => [...current, { id: newId(), role: "user", text: message }]);
    setInput("");
    setLoading(true);

    const result = await sendChat(message, sessionId);
    if (result.ok) {
      setSessionId(result.data.session_id);
      writeSession(result.data.session_id);
      setMessages((current) => [...current, { id: newId(), role: "assistant", response: result.data }]);
    } else {
      setMessages((current) => [...current, { id: newId(), role: "error", text: result.message }]);
    }
    setLoading(false);
    inputRef.current?.focus();
  }

  function newConversation() {
    setMessages([]);
    setSessionId(null);
    writeSession(null);
    setInput("");
    inputRef.current?.focus();
  }

  function onKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send(input);
    }
  }

  const tooLong = input.length > MAX_MESSAGE_CHARS;
  const canSend = input.trim().length > 0 && !tooLong && !loading;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1 className="header__title">Weather Safety Assistant</h1>
          <p className="header__subtitle">Live weather + written safety policies</p>
        </div>
        <button type="button" className="button button--ghost" onClick={newConversation} disabled={loading && messages.length === 0}>
          New conversation
        </button>
      </header>

      <main className="chat" aria-live="polite" aria-busy={loading}>
        {messages.length === 0 && (
          <div className="empty">
            <p className="empty__lead">Ask whether an outdoor plan is a good idea. Answers use live Open-Meteo weather and our written safety policies, and cite the policy used.</p>
            <p className="empty__hint">Try an example below.</p>
          </div>
        )}

        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}

        {loading && (
          <div className="message message--assistant">
            <div className="bubble bubble--assistant bubble--loading" role="status">
              <span className="dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
              Checking live weather and safety policies…
            </div>
          </div>
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer">
        {messages.length === 0 && (
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button key={example} type="button" className="chip-button" onClick={() => send(example)} disabled={loading}>
                {example}
              </button>
            ))}
          </div>
        )}
        <form
          className="composer__form"
          onSubmit={(event) => {
            event.preventDefault();
            send(input);
          }}
        >
          <label htmlFor="message" className="visually-hidden">
            Your question
          </label>
          <textarea
            id="message"
            ref={inputRef}
            className="composer__input"
            rows={1}
            placeholder="Ask about an outdoor plan…"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            disabled={loading}
            autoFocus
          />
          <button type="submit" className="button button--primary" disabled={!canSend}>
            {loading ? "Sending…" : "Send"}
          </button>
        </form>
        {tooLong && (
          <p className="composer__error" role="alert">
            Please shorten your question to {MAX_MESSAGE_CHARS} characters or fewer ({input.length} now).
          </p>
        )}
      </footer>
    </div>
  );
}
