type Column<T> = {
  key: string;
  header: string;
  render?: (row: T) => React.ReactNode;
  mono?: boolean;
};

type DataTableProps<T> = {
  columns: Column<T>[];
  rows: T[];
  emptyMessage?: string;
  getRowKey: (row: T) => string;
};

export function DataTable<T>({ columns, rows, emptyMessage = "No records yet.", getRowKey }: DataTableProps<T>) {
  if (rows.length === 0) {
    return <p className="research-empty">{emptyMessage}</p>;
  }

  return (
    <div className="research-table-wrap">
      <table className="research-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key} align="left">{col.header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={getRowKey(row)}>
              {columns.map((col) => (
                <td key={col.key} className={col.mono ? "research-table-mono" : undefined}>
                  {col.render
                    ? col.render(row)
                    : String((row as Record<string, unknown>)[col.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
