import { useState, useCallback } from "react";
import { useAppState } from "../app/appState";
import ViewSwitcher from "../components/topbar/ViewSwitcher";
import MetricsPanel from "../components/panels/MetricsPanel";
import Viewer from "../viewer/Viewer";
import { useFileUpload } from "../hooks/useFileUpload";
import DropZone from "../components/DropZone";

export default function ViewerScreen() {
  const { meshData, setMeshData, setPhase, phase } = useAppState();
  const [showDropZone, setShowDropZone] = useState(false);
  
  const { handleFileUpload, isUploading } = useFileUpload(setMeshData, setPhase);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      await handleFileUpload(file);
      setShowDropZone(false);
    }
  }, [handleFileUpload]);

  if (!meshData && phase !== "processing") {
    return (
      <div className="viewer-screen viewer-screen--empty">
        <div className="empty-state">
          <h2>Данные не загружены</h2>
          <button 
            className="upload-button"
            onClick={() => setShowDropZone(true)}
          >
            Загрузить файл
          </button>
        </div>
        
        {showDropZone && (
          <DropZone
            onDrop={onDrop}
            isUploading={isUploading}
            onClose={() => setShowDropZone(false)}
          />
        )}
      </div>
    );
  }

  if (phase === "processing" || isUploading) {
    return (
      <div className="viewer-screen viewer-screen--processing">
        <div className="processing-state">
          <h2>Обработка файла...</h2>
          <div className="spinner"></div>
          <p>Это может занять несколько минут</p>
        </div>
      </div>
    );
  }

  return (
    <div className="viewer-screen">
      <div className="sidebar-left">
        <div className="nav-section">
          <h3>Навигация</h3>
          <button 
            className="nav-button"
            onClick={() => setShowDropZone(true)}
          >
            Загрузить новый файл
          </button>
        </div>
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

      {showDropZone && (
        <DropZone
          onDrop={onDrop}
          isUploading={isUploading}
          onClose={() => setShowDropZone(false)}
        />
      )}
    </div>
  );
}