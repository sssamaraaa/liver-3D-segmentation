import { useAppState } from "../../app/appState";
import axes_icon from "../../../assets/axes.svg";
import threed_icon from "../../../assets/3D.svg";


export default function ViewSwitcher() {
  const { view, setView } = useAppState();
  
  const serverViews = ["3d", "axial", "sagittal", "coronal"];
  
  const russianLabels = {
    "3d": "3D",
    "axial": "Аксиальный",
    "sagittal": "Саггитальный",
    "coronal": "Корональный"
  };

  const svgIcons = {
    "3d": threed_icon,
    "axial": axes_icon,
    "sagittal": axes_icon,
    "coronal": axes_icon
  };

  return (
    <div className="switcher-container">
      {serverViews.map(v => (
        <button
          key={v}
          onClick={() => setView(v)} 
          className={`switcher-button ${view === v ? 'active' : ''}`}
        >
          <div className="button-content">
            <img 
              src={svgIcons[v]} 
              alt="" 
              className="switcher-icon"
            />
            <span>{russianLabels[v]}</span>
          </div>
        </button>
      ))}
    </div>
  );
}