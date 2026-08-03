import wordmarkUrl from "@/assets/seiso-wordmark.png";

/** Primary Seiso logo — detailed wordmark used everywhere in the UI. */
export function SeisoLogoMark({ className }: { className?: string }) {
  return (
    <img
      src={wordmarkUrl}
      alt="SEISO"
      className={className}
      draggable={false}
    />
  );
}

/** Alias kept for call sites that ask for the wordmark by name. */
export function SeisoWordmark({ className }: { className?: string }) {
  return <SeisoLogoMark className={className} />;
}
