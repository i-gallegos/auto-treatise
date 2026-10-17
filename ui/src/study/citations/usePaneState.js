import { useCallback, useMemo, useState } from "react";

export function usePaneState({
  colorBy: initialColorBy = "rank",
  treeOnly: initialTreeOnly = false,
  minCitedBy: initialMinCitedBy = 0,
} = {}) {
  const [colorBy, setColorByRaw] = useState(initialColorBy);
  // Cases with no leaf assignment are in the citation sweep but never entered
  // the induced tree. Off by default so the full graph stays visible.
  const [treeOnly, setTreeOnly] = useState(initialTreeOnly);
  // In-degree threshold: 0 or 1 means off. See citations/degreeFilter.js.
  const [minCitedBy, setMinCitedBy] = useState(initialMinCitedBy);
  const [hidden, setHidden] = useState(() => new Set());

  // Changing what the colours mean invalidates which groups were muted, so the
  // legend is cleared with it rather than carrying keys from another grouping.
  const setColorBy = useCallback((next) => {
    setColorByRaw(next);
    setHidden(new Set());
  }, []);

  const toggleHidden = useCallback((key) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  return useMemo(
    () => ({
      colorBy,
      setColorBy,
      treeOnly,
      setTreeOnly,
      minCitedBy,
      setMinCitedBy,
      hidden,
      toggleHidden,
    }),
    [
      colorBy, setColorBy, treeOnly, minCitedBy, hidden,
      toggleHidden,
    ],
  );
}
