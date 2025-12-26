import { useAppState } from "../../app/appState";

export default function ViewSwitcher() {
  const { view, setView } = useAppState();
  const serverViews = ["3d", "axial", "sagittal", "coronal"];
  

  const russianLabels = {
    "3d": "3D",
    "axial": "Аксиальный",
    "sagittal": "Саггитальный",
    "coronal": "Корональный"
  };

  return (
    <div className="switcher-container">
      {serverViews.map(v => (
        <button
          key={v}
          onClick={() => setView(v)} 
          className={`switcher-button ${view === v ? 'active' : ''}`}
        >
          {russianLabels[v]} 
        </button>
      ))}
    </div>
  );
}