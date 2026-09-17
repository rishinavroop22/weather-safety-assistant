import CheckedPanel from "./CheckedPanel.jsx";

// Neutral labels for the backend's status/reason codes. Display only: they describe
// what happened to the request, never whether an activity is safe.
const REASON_LABELS = {
  llm_unavailable: "Assistant unavailable",
  weather_error: "Weather service unavailable",
  location_error: "Location lookup unavailable",
  location_not_found: "Location not found",
  location_missing: "Needs a location",
  activity_missing: "Needs more detail",
  window_passed: "Time has passed",
  invalid_intent: "Couldn't understand",
  unsupported_request: "Outside what I can help with",
  no_sop: "No guidance available",
  verification_failed: "Policy guidance",
};
const STATUS_LABELS = {
  answered: "Policy-based answer",
  answered_fallback: "Policy guidance",
  no_guidance: "No guidance available",
  unsupported: "Outside what I can help with",
  clarification: "Needs more detail",
  failure: "Couldn't complete",
};
const STATUS_TONE = {
  answered: "ok",
  answered_fallback: "ok",
  no_guidance: "neutral",
  unsupported: "neutral",
  clarification: "neutral",
  failure: "warn",
};

export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="message message--user">
        <div className="bubble bubble--user">{message.text}</div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="message message--assistant" role="alert">
        <div className="bubble bubble--error">
          <span className="chip chip--warn">Request failed</span>
          <p className="bubble__text">{message.text}</p>
        </div>
      </div>
    );
  }

  const response = message.response;
  const label = REASON_LABELS[response.reason] ?? STATUS_LABELS[response.status] ?? "Answer";
  const tone = STATUS_TONE[response.status] ?? "neutral";
  return (
    <div className="message message--assistant">
      <div className={`bubble bubble--assistant bubble--${tone}`}>
        <span className={`chip chip--${tone}`}>{label}</span>
        <p className="bubble__text">{response.answer_text || response.answer}</p>
        <CheckedPanel response={response} />
      </div>
    </div>
  );
}
