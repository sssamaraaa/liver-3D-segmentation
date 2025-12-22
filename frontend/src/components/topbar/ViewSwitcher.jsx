import { useAppState } from "../../app/appState";

export default function ViewSwitcher() {
  const { view, setView } = useAppState();
  const views = ["3d", "axial", "sagittal", "coronal"];

  return (
    <div className="flex gap-2 p-2 bg-zinc-800">
      {views.map(v => (
        <button
          key={v}
          onClick={() => setView(v)}
          className={`px-3 py-1 rounded ${
            view === v ? "bg-emerald-600" : "bg-zinc-700"
          }`}
        >
          {v.toUpperCase()}
        </button>
      ))}
    </div>
  );
}
