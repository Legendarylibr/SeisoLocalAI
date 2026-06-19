import { useEffect, useRef, useState } from "react";

/** Shared open/search/focus/outside-dismiss behavior for Hub combobox pickers. */
export function useHubCombobox(resetOnOpen = true) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    if (resetOnOpen) setSearch("");
    requestAnimationFrame(() => searchRef.current?.focus());
  }, [open, resetOnOpen]);

  return { open, setOpen, search, setSearch, rootRef, searchRef };
}
