import { Link } from "react-router-dom";
import type { FC } from "react";
import { IconChevronRight } from "@/components/Icons";

type Pipeline = {
  id: string;
  title: string;
  desc: string;
  path: string;
  tag?: string;
  Icon: FC<{ size?: number }>;
};

export function PipelineStrip({ pipelines }: { pipelines: Pipeline[] }) {
  return (
    <div className="pipeline-strip">
      {pipelines.map((p) => (
        <Link key={p.id} to={p.path} className="pipeline-card">
          <div className="pipeline-card-icon">
            <p.Icon size={18} />
          </div>
          <div className="pipeline-card-body">
            <div className="pipeline-card-head">
              <span className="pipeline-card-title">{p.title}</span>
              {p.tag && <span className="pipeline-card-tag">{p.tag}</span>}
            </div>
            <p className="pipeline-card-desc">{p.desc}</p>
          </div>
          <IconChevronRight size={14} className="pipeline-card-arrow" />
        </Link>
      ))}
    </div>
  );
}
