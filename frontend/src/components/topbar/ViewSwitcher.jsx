import { useAppState } from "../../app/appState";

export default function ViewSwitcher() {
  const { view, setView } = useAppState();
  const views = ["3D", "Аксиальный", "Саггитальный", "Корональный"];

  return (
    <div className="switcher-container">
      {views.map(v => (
        <button
          key={v}
          onClick={() => setView(v)}
          className={`switcher-button ${view === v ? 'active' : ''}`}
        >
          {v}
        </button>
      ))}
    </div>
  );
}
