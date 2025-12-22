import { useAppState } from "../app/appState";
import ViewSwitcher from "../components/topbar/ViewSwitcher";
import MetricsPanel from "../components/panels/MetricsPanel";
import Viewer from "../viewer/Viewer";

export default function ViewerScreen() {
  const { meshData } = useAppState();

  return (
    <div className="viewer-screen">
      <div className="sidebar-left">
        Навигация
      </div>

      <div className="main-area">
        <div className="top-bar">
          <ViewSwitcher />
        </div>
        
        <div className="viewer-container">
          <Viewer />
        </div>
      </div>

      <div className="sidebar-right">
        <MetricsPanel metrics={meshData?.metrics} />
      </div>
    </div>
  );
}