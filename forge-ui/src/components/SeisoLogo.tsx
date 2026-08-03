import mascotUrl from "@/assets/seiso-mascot.png";
import wordmarkUrl from "@/assets/seiso-wordmark.png";

export function SeisoLogoMark({ className }: { className?: string }) {
  return <img src={mascotUrl} alt="" className={className} draggable={false} />;
}

/** Full detailed SEISO wordmark (warm glitch style). */
export function SeisoWordmark({ className }: { className?: string }) {
  return (
    <img
      src={wordmarkUrl}
      alt="SEISO"
      className={className}
      draggable={false}
    />
  );
}
