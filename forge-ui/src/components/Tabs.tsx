import { useId } from "react";

type TabItem<T extends string> = {
  id: T;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  count?: number | string;
  badge?: string;
};

type TabsProps<T extends string> = {
  items: TabItem<T>[];
  value: T;
  onChange: (id: T) => void;
  variant?: "segmented" | "underline";
  className?: string;
  "aria-label"?: string;
};

export function Tabs<T extends string>({
  items,
  value,
  onChange,
  variant = "segmented",
  className = "",
  "aria-label": ariaLabel = "Sections",
}: TabsProps<T>) {
  const baseId = useId();

  return (
    <div
      className={`tab-bar tab-bar-${variant}${className ? ` ${className}` : ""}`}
      role="tablist"
      aria-label={ariaLabel}
    >
      {items.map((item) => {
        const active = item.id === value;
        const tabId = `${baseId}-${item.id}`;

        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={tabId}
            aria-selected={active}
            aria-controls={`${tabId}-panel`}
            className={`tab-item${active ? " tab-item-active" : ""}`}
            onClick={() => onChange(item.id)}
          >
            {item.icon && <span className="tab-item-icon">{item.icon}</span>}
            <span className="tab-item-body">
              <span className="tab-item-label-row">
                <span className="tab-item-label">{item.label}</span>
                {item.count != null && (
                  <span className="tab-item-count">{item.count}</span>
                )}
                {item.badge && <span className="tab-item-badge">{item.badge}</span>}
              </span>
              {item.description && (
                <span className="tab-item-desc">{item.description}</span>
              )}
            </span>
            {active && variant === "segmented" && (
              <span className="tab-item-indicator" aria-hidden />
            )}
          </button>
        );
      })}
    </div>
  );
}
