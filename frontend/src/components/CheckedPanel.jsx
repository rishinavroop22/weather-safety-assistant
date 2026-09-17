// Shows the metadata the backend returned with an answer. It displays values only;
// it never computes or interprets weather or safety.

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

function retrievedTime(isoUtc) {
  if (!isoUtc) return null;
  const date = new Date(isoUtc);
  return Number.isNaN(date.getTime()) ? null : `${date.toISOString().slice(11, 16)} UTC`;
}

export default function CheckedPanel({ response }) {
  const { location, time_window: window, policy, weather_summary: weather } = response;
  if (!location && !window && !policy && !weather) return null;

  const retrieved = retrievedTime(weather?.retrieved_at_utc);

  return (
    <section className="checked" aria-label="What was checked">
      <h3 className="checked__title">Checked</h3>
      <dl className="checked__grid">
        {location && (
          <div className="checked__item">
            <dt>Location</dt>
            <dd>{location}</dd>
          </div>
        )}
        {window && (
          <div className="checked__item">
            <dt>Time</dt>
            <dd>{capitalize(window.description)}</dd>
          </div>
        )}
        {policy && (
          <div className="checked__item">
            <dt>Policy</dt>
            <dd>
              {policy.sop_ids.join(" + ")} · {policy.severity} severity
              <span className="checked__sub">{policy.primary.title}</span>
              {policy.also_applies.map((sop) => (
                <span className="checked__sub" key={sop.id}>
                  Also: {sop.title} ({sop.severity})
                </span>
              ))}
            </dd>
          </div>
        )}
        {weather && (weather.values.length > 0 || weather.unavailable.length > 0) && (
          <div className="checked__item checked__item--wide">
            <dt>Weather checked</dt>
            <dd>
              <ul className="checked__values">
                {weather.values.map((item) => (
                  <li key={item.label}>
                    <span>{capitalize(item.label)}</span>
                    <strong>{item.value}</strong>
                  </li>
                ))}
                {weather.unavailable.map((label) => (
                  <li key={label} className="checked__unavailable">
                    <span>{capitalize(label)}</span>
                    <strong>not available</strong>
                  </li>
                ))}
              </ul>
              <span className="checked__sub">
                Source: {weather.source}
                {retrieved ? ` · retrieved ${retrieved}` : ""}
              </span>
            </dd>
          </div>
        )}
      </dl>
    </section>
  );
}
