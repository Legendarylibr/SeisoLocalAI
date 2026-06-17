import { useState } from "react";
import { IconChevronDown } from "@/components/Icons";

type FormSectionProps = {
  title: string;
  hint?: string;
  defaultOpen?: boolean;
  collapsible?: boolean;
  children: React.ReactNode;
};

export function FormSection({
  title,
  hint,
  defaultOpen = true,
  collapsible = false,
  children,
}: FormSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  if (!collapsible) {
    return (
      <section className="form-section">
        <div className="form-section-head">
          <h3 className="form-section-title">{title}</h3>
          {hint && <p className="form-section-hint">{hint}</p>}
        </div>
        <div className="form-section-body">{children}</div>
      </section>
    );
  }

  return (
    <section className={`form-section form-section-collapsible${open ? " form-section-open" : ""}`}>
      <button
        type="button"
        className="form-section-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="form-section-toggle-text">
          <h3 className="form-section-title">{title}</h3>
          {hint && <p className="form-section-hint">{hint}</p>}
        </span>
        <IconChevronDown size={16} className="form-section-chevron" />
      </button>
      {open && <div className="form-section-body">{children}</div>}
    </section>
  );
}
