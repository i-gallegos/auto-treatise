import { overlayDisplayName } from "./labelOverlay";

export function LabelOverlaySelect({
  value = "",
  options = [],
  onChange,
}) {
  return (
    <label className="study-field">
      <span className="study-field-label">Labels</span>
      <select
        className="study-field-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Tree label overlay"
      >
        <option value="">Original</option>
        {options.map((opt) => (
          <option key={opt.tag} value={opt.tag}>
            {overlayDisplayName(opt)}
          </option>
        ))}
      </select>
    </label>
  );
}
