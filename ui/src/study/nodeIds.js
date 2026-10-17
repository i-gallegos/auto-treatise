export function displayNodeId(id) {
  if (id === null || id === undefined) return id;
  return String(id).replace(/^n0(?=\.|$)/, "v");
}
