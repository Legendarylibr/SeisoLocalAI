import { useCallback, useState } from "react";
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

function RecipeNode({ data }: NodeProps) {
  return (
    <div className="recipe-node">
      <Handle type="target" position={Position.Left} />
      <strong>{data.label as string}</strong>
      <div className="recipe-node-type">{data.nodeType as string}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { recipe: RecipeNode };

const INITIAL_NODES: Node[] = [
  { id: "import", type: "recipe", position: { x: 0, y: 80 }, data: { label: "Import", nodeType: "import" } },
  { id: "transform", type: "recipe", position: { x: 220, y: 80 }, data: { label: "Transform", nodeType: "transform" } },
  { id: "filter", type: "recipe", position: { x: 440, y: 80 }, data: { label: "Filter", nodeType: "filter" } },
  { id: "sample", type: "recipe", position: { x: 660, y: 80 }, data: { label: "Sample", nodeType: "sample" } },
  { id: "out", type: "recipe", position: { x: 880, y: 80 }, data: { label: "Output", nodeType: "output" } },
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

type Props = {
  onChange?: (recipe: RecipeGraph) => void;
};

export function RecipeCanvas({ onChange }: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState(INITIAL_NODES);
  const [edges, setEdges, onEdgesChange] = useEdgesState(INITIAL_EDGES);
  const [name, setName] = useState("Custom recipe");

  const onConnect = useCallback(
    (conn: Connection) => setEdges((eds) => addEdge(conn, eds)),
    [setEdges],
  );

  const addNode = (type: string) => {
    const id = `${type}_${Date.now()}`;
    setNodes((nds) => [
      ...nds,
      {
        id,
        type: "recipe",
        position: { x: 100 + nds.length * 40, y: 200 + nds.length * 30 },
        data: { label: type.charAt(0).toUpperCase() + type.slice(1), nodeType: type },
      },
    ]);
  };

  const toRecipe = (): RecipeGraph => {
    const recipeNodes = nodes.map((n) => ({
      id: n.id,
      type: (n.data.nodeType as string) || "transform",
      config: buildConfig(n.id, (n.data.nodeType as string) || "transform", nodes, edges),
    }));
    const recipeEdges = edges.map((e) => ({ source: e.source, target: e.target }));
    const graph = { name, nodes: recipeNodes, edges: recipeEdges };
    onChange?.(graph);
    return graph;
  };

  return (
    <div className="recipe-canvas-wrap">
      <div className="recipe-toolbar">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Recipe name" />
        {NODE_TYPES_LIST.map((t) => (
          <button key={t} type="button" className="btn" onClick={() => addNode(t)}>
            + {t}
          </button>
        ))}
        <button type="button" className="btn btn-primary" onClick={() => toRecipe()}>
          Serialize
        </button>
      </div>
      <div className="recipe-flow">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
        >
          <Background />
          <Controls />
          <MiniMap />
        </ReactFlow>
      </div>
    </div>
  );
}

function buildConfig(
  nodeId: string,
  nodeType: string,
  _nodes: Node[],
  edges: Edge[],
): Record<string, unknown> {
  const incoming = edges.find((e) => e.target === nodeId);
  const source = incoming?.source;

  switch (nodeType) {
    case "import":
      return { path: "data/input.txt", format: "txt" };
    case "transform":
      return { source, template: "Instruction: {text}\nOutput:" };
    case "filter":
      return { source, min_length: 20 };
    case "sample":
      return { source, count: 100 };
    case "output":
      return { source };
    default:
      return { source };
  }
}

export function recipeFromCanvas(name: string, nodes: Node[], edges: Edge[]): RecipeGraph {
  return {
    name,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: (n.data.nodeType as string) || "transform",
      config: buildConfig(n.id, (n.data.nodeType as string) || "transform", nodes, edges),
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  };
}
