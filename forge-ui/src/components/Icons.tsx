import type { FC, ReactNode } from "react";

type IconProps = { size?: number; className?: string; strokeWidth?: number };

const defaults = { size: 18, strokeWidth: 1.65 };

function Svg({ size, className, children }: { size: number; className?: string; children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden
    >
      {children}
    </svg>
  );
}

export function IconDashboard({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="13.5" y="3.5" width="7" height="5" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="13.5" y="11.5" width="7" height="9" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
    </Svg>
  );
}

export function IconHub({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="12" cy="12" r="2.25" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="5" cy="7" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="19" cy="7" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="5" cy="17" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="19" cy="17" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M6.8 8.2L10 10.5M17.2 8.2L14 10.5M6.8 15.8L10 13.5M17.2 15.8L14 13.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconChat({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path
        d="M5.5 5.5H18.5C19.6 5.5 20.5 6.4 20.5 7.5V14.5C20.5 15.6 19.6 16.5 18.5 16.5H9.5L5.5 19.5V16.5H5.5C4.4 16.5 3.5 15.6 3.5 14.5V7.5C3.5 6.4 4.4 5.5 5.5 5.5Z"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />
      <path d="M8.5 10H15.5M8.5 13H12.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconTrain({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M4.5 14.5V9.5L12 5.5L19.5 9.5V14.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M8 14.5V11.5L12 9.5L16 11.5V14.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinejoin="round" />
      <path d="M4.5 14.5H19.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M9 18.5H15" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconCompress({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M8.5 4.5L4.5 8.5M4.5 8.5H8M4.5 8.5V4.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M15.5 19.5L19.5 15.5M19.5 15.5H16M19.5 15.5V19.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <rect x="8.5" y="8.5" width="7" height="7" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
    </Svg>
  );
}

export function IconExport({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M12 4.5V14.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M8.5 11L12 14.5L15.5 11" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5.5 18.5H18.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconRecipes({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="4.5" y="4.5" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="13.5" y="4.5" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="4.5" y="13.5" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="13.5" y="13.5" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M10.5 7.5H13.5M7.5 10.5V13.5M16.5 10.5V13.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconIntegrations({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M9 7.5H6.5C5.4 7.5 4.5 8.4 4.5 9.5V11C4.5 12.1 5.4 13 6.5 13H9" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M15 7.5H17.5C18.6 7.5 19.5 8.4 19.5 9.5V11C19.5 12.1 18.6 13 17.5 13H15" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M9 12H15" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="12" cy="12" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
    </Svg>
  );
}

export function IconShield({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path
        d="M12 3.5L19 6.75V12.25C19 16.25 16.25 19.25 12 20.75C7.75 19.25 5 16.25 5 12.25V6.75L12 3.5Z"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />
      <path d="M9.25 12L11 13.75L14.75 10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconMenu({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M5 7H19M5 12H19M5 17H19" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconClose({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M7 7L17 17M17 7L7 17" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconSearch({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="11" cy="11" r="5.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M15.5 15.5L19 19" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconChevronDown({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M7 10L12 15L17 10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconChevronRight({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M10 7L15 12L10 17" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconChevronLeft({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M14 7L9 12L14 17" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconSend({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M12 19V11" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M8.5 11.5L12 8L15.5 11.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconRefresh({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M18 8.5V4.5H14" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6 15.5V19.5H10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M18.5 10.5C17.4 7.2 14.3 5 11 5C7.1 5 4 8.1 4 12C4 15.9 7.1 19 11 19C14.1 19 16.8 17.1 18 14.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconEject({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M12 4.5V13.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M8.5 10L12 13.5L15.5 10" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6.5 17.5H17.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconLock({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="6.5" y="10.5" width="11" height="8" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M8.5 10.5V8C8.5 6.07 10.07 4.5 12 4.5C13.93 4.5 15.5 6.07 15.5 8V10.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconActivity({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M4.5 12H7.5L9.5 6.5L12.5 17.5L14.5 12H19.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

export function IconPlus({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path d="M12 6.5V17.5M6.5 12H17.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconCpu({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="7.5" y="7.5" width="9" height="9" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M9.5 4.5V7.5M12 4.5V7.5M14.5 4.5V7.5M9.5 16.5V19.5M12 16.5V19.5M14.5 16.5V19.5M4.5 9.5H7.5M4.5 12H7.5M4.5 14.5H7.5M16.5 9.5H19.5M16.5 12H19.5M16.5 14.5H19.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconMemory({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="4.5" y="7" width="15" height="10" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M7.5 7V5.5M10.5 7V5.5M13.5 7V5.5M16.5 7V5.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M7.5 17V18.5M10.5 17V18.5M13.5 17V18.5M16.5 17V18.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconGpu({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="3.5" y="8.5" width="17" height="9" rx="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M7.5 12H10.5M7.5 14.5H10.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="15.5" cy="13" r="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M8.5 8.5V6.5M15.5 8.5V6.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconInference({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M12 3.5V6.5M12 17.5V20.5M20.5 12H17.5M6.5 12H3.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M17.8 6.2L15.7 8.3M8.3 15.7L6.2 17.8M17.8 17.8L15.7 15.7M8.3 8.3L6.2 6.2" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconAssistant({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="12" cy="12" r="7.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="9" cy="11" r="1" fill="currentColor" />
      <circle cx="15" cy="11" r="1" fill="currentColor" />
      <path d="M9.5 14.5C10.3 15.5 11.4 16 12 16C12.6 16 13.7 15.5 14.5 14.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconGlobe({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="12" cy="12" r="7.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M4.5 12H19.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <path d="M12 4.5C9.8 7.8 9.8 16.2 12 19.5C14.2 16.2 14.2 7.8 12 4.5Z" stroke="currentColor" strokeWidth={strokeWidth} strokeLinejoin="round" />
    </Svg>
  );
}

export function IconHardDrive({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="4.5" y="6.5" width="15" height="11" rx="2" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M4.5 11H19.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="8" cy="14.5" r="0.85" fill="currentColor" />
      <circle cx="11" cy="14.5" r="0.85" fill="currentColor" />
    </Svg>
  );
}

export function IconServer({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <rect x="4.5" y="5" width="15" height="5.5" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <rect x="4.5" y="13.5" width="15" height="5.5" rx="1.5" stroke="currentColor" strokeWidth={strokeWidth} />
      <circle cx="8" cy="7.75" r="0.75" fill="currentColor" />
      <circle cx="8" cy="16.25" r="0.75" fill="currentColor" />
    </Svg>
  );
}

export function IconUser({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <circle cx="12" cy="8.5" r="3" stroke="currentColor" strokeWidth={strokeWidth} />
      <path d="M6 18.5C6.8 15.2 9.1 13.5 12 13.5C14.9 13.5 17.2 15.2 18 18.5" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
    </Svg>
  );
}

export function IconKnowledge({ size = defaults.size, className, strokeWidth = defaults.strokeWidth }: IconProps) {
  return (
    <Svg size={size} className={className}>
      <path
        d="M5.5 5.5H18.5C19.6 5.5 20.5 6.4 20.5 7.5V16.5C20.5 17.6 19.6 18.5 18.5 18.5H5.5C4.4 18.5 3.5 17.6 3.5 16.5V7.5C3.5 6.4 4.4 5.5 5.5 5.5Z"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />
      <path d="M8 8.5H16M8 12H14M8 15.5H12" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" />
      <circle cx="16.5" cy="14" r="2.5" stroke="currentColor" strokeWidth={strokeWidth} />
    </Svg>
  );
}

export type NavIconName =
  | "dashboard"
  | "hub"
  | "chat"
  | "knowledge"
  | "train"
  | "compress"
  | "export"
  | "recipes"
  | "integrations";

const NAV_ICONS: Record<NavIconName, FC<IconProps>> = {
  dashboard: IconDashboard,
  hub: IconHub,
  chat: IconChat,
  knowledge: IconKnowledge,
  train: IconTrain,
  compress: IconCompress,
  export: IconExport,
  recipes: IconRecipes,
  integrations: IconIntegrations,
};

export function NavIcon({ name, ...props }: IconProps & { name: NavIconName }) {
  const C = NAV_ICONS[name];
  return <C {...props} />;
}
