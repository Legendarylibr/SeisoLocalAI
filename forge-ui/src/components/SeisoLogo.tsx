import mascotPngUrl from "@/assets/seiso-mascot.png";
import mascotWebpUrl from "@/assets/seiso-mascot.webp";

export function SeisoLogoMark({ className }: { className?: string }) {
  return (
    <picture>
      <source srcSet={mascotWebpUrl} type="image/webp" />
      <img src={mascotPngUrl} alt="" className={className} draggable={false} />
    </picture>
  );
}
