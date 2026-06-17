import mascotUrl from "@/assets/seiso-mascot.png";

export { mascotUrl };

export function SeisoLogoMark({ className }: { className?: string }) {
  return <img src={mascotUrl} alt="" className={className} draggable={false} />;
}
