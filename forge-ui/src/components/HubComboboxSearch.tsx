import { IconClose, IconSearch } from "@/components/Icons";

type HubComboboxSearchProps = {
  searchRef: React.RefObject<HTMLInputElement | null>;
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
  onEscape?: () => void;
  onEnter?: () => void;
};

export function HubComboboxSearch({
  searchRef,
  value,
  placeholder,
  onChange,
  onEscape,
  onEnter,
}: HubComboboxSearchProps) {
  return (
    <div className="chat-model-picker-search">
      <span className="chat-model-picker-search-icon" aria-hidden>
        <IconSearch size={15} />
      </span>
      <input
        ref={searchRef}
        type="search"
        className="chat-model-picker-search-input"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            onEscape?.();
          }
          if (e.key === "Enter") {
            e.preventDefault();
            onEnter?.();
          }
        }}
      />
      {value && (
        <button
          type="button"
          className="chat-model-picker-search-clear"
          onClick={() => onChange("")}
          aria-label="Clear search"
        >
          <IconClose size={14} />
        </button>
      )}
    </div>
  );
}
