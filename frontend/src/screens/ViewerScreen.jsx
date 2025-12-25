import { useState, useCallback } from "react";
import { useAppState } from "../app/appState";
import Viewer from "../viewer/Viewer";
import AppLayout from "../app/AppLayout";
import DropZone from "../components/DropZone";
import { useFileUpload } from "../hooks/useFileUpload";
import MetricsPanel from "../components/panels/MetricsPanel";
import ViewSwitcher from "../components/topbar/ViewSwitcher";
import ExportPanel from "../components/panels/ExportPanel";

export default function ViewerScreen() {
  const { meshData } = useAppState();
  const { handleFileUpload } = useFileUpload();
  const [showDropZone, setShowDropZone] = useState(false);

  const onDrop = useCallback(files => {
    handleFileUpload(files[0]);
    setShowDropZone(false);
  }, []);

  function handleExport(type) {
    console.log("meshData:", meshData);
    console.log("meshData.outputs:", meshData?.outputs);
    console.log("meshData.outputs?.mask:", meshData?.outputs?.mask);
    if (!meshData?.outputs && !meshData?.files) return;

    const fileMap = {
      stl: meshData.outputs?.mesh_stl || meshData.files?.mesh_stl,
      ply: meshData.outputs?.mesh_ply || meshData.files?.mesh_ply,
      nii: meshData.files?.mask?.nifti || meshData.mask_path,
      json: meshData.outputs?.report_json || meshData.files?.report_json
    };

    const filePath = fileMap[type];
    if (!filePath) {
      console.warn("Файл для экспорта отсутствует:", type);
      return;
    }

    let fullUrl = filePath;
    if (!filePath.startsWith("http")) {
      const baseUrl = "http://localhost:8000";
      fullUrl = `${baseUrl}${filePath.startsWith("/") ? filePath : "/" + filePath}`;
    }

    fetch(fullUrl)
      .then(response => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.blob();
      })
      .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        
        const fileName = filePath.split("/").pop() || 
                        `${meshData?.patient_id || 'export'}_${new Date().toISOString().split('T')[0]}.${type}`;
        link.download = fileName;
        
        document.body.appendChild(link);
        link.click();
        
        setTimeout(() => {
          document.body.removeChild(link);
          window.URL.revokeObjectURL(url);
        }, 100);
      })
      .catch(error => {
        console.error('Ошибка при скачивании:', error);
        window.open(fullUrl, '_blank');
      });
  }

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
            <ExportPanel onExport={handleExport} />  
          </div>

          <div className="main-area">
            <div className="view-switcher-container">
              <ViewSwitcher />
            </div>

            <div className="viewer-container">
              <Viewer />
            </div>
          </div>

          <div className="sidebar-right">
            <MetricsPanel metrics={meshData?.metrics} />
          </div>

          {showDropZone && <DropZone onDrop={onDrop} onClose={() => setShowDropZone(false)} />}
        </div>
      )}
    </AppLayout>
  );
}
