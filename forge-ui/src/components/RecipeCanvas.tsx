import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  Connection,
  Node,
  Edge,
  Handle,
  Position,
  NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

const NODE_TYPES_LIST = ["import", "transform", "filter", "sample", "output"] as const;

const NODE_HELP: Record<string, string> = {
  import: "Read a text or JSONL file from your uploads folder.",
  transform: "Apply an instruction template to each row.",
  filter: "Drop rows shorter than a minimum length.",
  sample: "Randomly sample up to N rows.",
  output: "Write the final dataset as JSONL.",
};

const PIPELINE_STEPS = [
  { step: 1, title: "Import", detail: "Point to a file under your uploads directory." },
  { step: 2, title: "Transform & filter", detail: "Shape raw text into training-ready examples." },
  { step: 3, title: "Sample & export", detail: "Cap dataset size and write JSONL output." },
];

function RecipeNode({ data, selected }: NodeProps) {
  return (
    <div className={`recipe-node${selected ? " recipe-node-selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <strong>{data.label as string}</strong>
      <div className="recipe-node-type">{data.nodeType as string}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { recipe: RecipeNode };

const INITIAL_NODES: Node[] = [
  { id: "import", type: "recipe", position: { x: 0, y: 100 }, data: { label: "Import", nodeType: "import" } },
  { id: "transform", type: "recipe", position: { x: 240, y: 100 }, data: { label: "Transform", nodeType: "transform" } },
  { id: "filter", type: "recipe", position: { x: 480, y: 100 }, data: { label: "Filter", nodeType: "filter" } },
  { id: "sample", type: "recipe", position: { x: 720, y: 100 }, data: { label: "Sample", nodeType: "sample" } },
  { id: "out", type: "recipe", position: { x: 960, y: 100 }, data: { label: "Output", nodeType: "output" } },
];

const INITIAL_EDGES: Edge[] = [
  { id: "e1", source: "import", target: "transform" },
  { id: "e2", source: "transform", target: "filter" },
  { id: "e3", source: "filter", target: "sample" },
  { id: "e4", source: "sample", target: "out" },
];

export type RecipeGraph = {
  name: string;
  nodes: Array<{ id: string; type: string; config: Record<string, unknown> }>;
  edges: Array<{ source: string; target: string }>;
};

type NodeConfig = {
  importPath: string;
  transformTemplate: string;
  filterMinLength: number;
  sampleCount: number;
};

type Props = {
  onChange?: (recipe: RecipeGraph) => void;
};

export function RecipeCanvas({ onChange }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState(INITIAL_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES);
  const [name, setName] = useState("Custom recipe");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>("import");
  const [config, setConfig] = useState<NodeConfig>({
    importPath: "",
    transformTemplate: "Instruction: {text}\nOutput:",
    filterMinLength: 20,
    sampleCount: 100,
  });

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId) ?? null,
    [nodes, selectedNodeId],
  );
  const selectedType = (selectedNode?.data.nodeType as string) || "import";

  const onConnect = useCallback(
    (conn: Connection) => setEdges((eds) => addEdge(conn, eds)),
    [setEdges],
  );

  const toRecipe = useCallback((): RecipeGraph => {
    const recipeNodes = nodes.map((n) => ({
      id: n.id,
      type: (n.data.nodeType as string) || "transform",
      config: buildConfig(n.id, (n.data.nodeType as string) || "transform", edges, config),
    }));
    const recipeEdges = edges.map((e) => ({ source: e.source, target: e.target }));
    return { name, nodes: recipeNodes, edges: recipeEdges };
  }, [nodes, edges, name, config]);

  useEffect(() => {
    onChange?.(toRecipe());
  }, [onChange, toRecipe]);

  const addNode = (type: string) => {
    const id = `${type}_${Date.now()}`;
    setNodes((nds) => [
      ...nds,
      {
        id,
        type: "recipe",
        position: { x: 120 + nds.length * 48, y: 240 + nds.length * 24 },
        data: { label: type.charAt(0).toUpperCase() + type.slice(1), nodeType: type },
      },
    ]);
    setSelectedNodeId(id);
  };

  const updateConfig = <K extends keyof NodeConfig>(key: K, value: NodeConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="recipe-studio-layout">
      <aside className="recipe-sidebar">
        <div className="recipe-sidebar-section">
          <label className="recipe-field-label" htmlFor="recipe-name">Recipe name</label>
          <input id="recipe-name" value={name} onChange={(e) => setName(e.target.value)} placeholder="My dataset pipeline" />
        </div>

        <div className="recipe-sidebar-section">
          <span className="recipe-field-label">Pipeline steps</span>
          <ol className="recipe-step-list">
            {PIPELINE_STEPS.map((s) => (
              <li key={s.step}>
                <strong>{s.title}</strong>
                <span>{s.detail}</span>
              </li>
            ))}
          </ol>
        </div>

        <div className="recipe-sidebar-section">
          <span className="recipe-field-label">Add node</span>
          <div className="recipe-node-palette">
            {NODE_TYPES_LIST.map((t) => (
              <button key={t} type="button" className="btn btn-sm" onClick={() => addNode(t)}>
                + {t}
              </button>
            ))}
          </div>
        </div>

        {selectedNode && (
          <div className="recipe-sidebar-section recipe-node-config">
            <span className="recipe-field-label">Configure: {selectedType}</span>
            <p className="muted-text recipe-node-help">{NODE_HELP[selectedType]}</p>
            {selectedType === "import" && (
              <label>
                File path
                <input
                  value={config.importPath}
                  onChange={(e) => updateConfig("importPath", e.target.value)}
                  placeholder="/path/to/uploads/my-data.txt"
                />
              </label>
            )}
            {selectedType === "transform" && (
              <label>
                Template
                <textarea
                  rows={4}
                  value={config.transformTemplate}
                  onChange={(e) => updateConfig("transformTemplate", e.target.value)}
                />
              </label>
            )}
            {selectedType === "filter" && (
              <label>
                Min length
                <input
                  type="number"
                  min={1}
                  value={config.filterMinLength}
                  onChange={(e) => updateConfig("filterMinLength", Number(e.target.value) || 1)}
                />
              </label>
            )}
            {selectedType === "sample" && (
              <label>
                Row count
                <input
                  type="number"
                  min={1}
                  value={config.sampleCount}
                  onChange={(e) => updateConfig("sampleCount", Number(e.target.value) || 1)}
                />
              </label>
            )}
          </div>
        )}
      </aside>

      <div className="recipe-canvas-wrap">
        <div className="recipe-flow">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            nodeTypes={nodeTypes}
            fitView
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

function buildConfig(
  nodeId: string,
  nodeType: string,
  edges: Edge[],
  config: NodeConfig,
): Record<string, unknown> {
  const incoming = edges.find((e) => e.target === nodeId);
  const source = incoming?.source;

  switch (nodeType) {
    case "import":
      return { path: config.importPath, format: "txt" };
    case "transform":
      return { source, template: config.transformTemplate };
    case "filter":
      return { source, min_length: config.filterMinLength };
    case "sample":
      return { source, count: config.sampleCount };
    case "output":
      return { source };
    default:
      return { source };
  }
}
