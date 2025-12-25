import AppLayout from "../app/AppLayout";
import { useState, useCallback } from "react";
import { useAppState } from "../app/appState";
import Viewer from "../viewer/Viewer";
import DropZone from "../components/DropZone";
import { useFileUpload } from "../hooks/useFileUpload";
import MetricsPanel from "../components/panels/MetricsPanel";
import ViewSwitcher from "../components/topbar/ViewSwitcher";

export default function ViewerScreen() {
  const { meshData } = useAppState();
  const { handleFileUpload } = useFileUpload();
  const [showDropZone, setShowDropZone] = useState(false);

  const onDrop = useCallback(files => {
    handleFileUpload(files[0]);
    setShowDropZone(false);
  }, []);

  return (
    <AppLayout>
      {!meshData ? (
        <div className="viewer-screen viewer-screen--empty">
          <h2>Данные не загружены</h2>
          <button onClick={() => setShowDropZone(true)}>Загрузить файл</button>
          {showDropZone && <DropZone onDrop={onDrop} onClose={() => setShowDropZone(false)} />}
        </div>
      ) : (
        <div className="viewer-screen">

          <div className="sidebar-left">
            <button onClick={() => setShowDropZone(true)}>Загрузить новый файл</button>
          </div>

          <div className="main-area">
            <ViewSwitcher />
            <Viewer />
          </div>

          <div className="sidebar-right">
            <MetricsPanel metrics={meshData?.metrics}/>
          </div>

          {showDropZone && <DropZone onDrop={onDrop} onClose={() => setShowDropZone(false)} />}
        </div>
      )}
    </AppLayout>
  );
}
