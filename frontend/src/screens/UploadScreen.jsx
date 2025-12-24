import { useAppState } from "../app/appState";
import { useFileUpload } from "../hooks/useFileUpload";

export default function UploadScreen() {
  const { setPhase, setMeshData } = useAppState();
  const { handleFileUpload, isUploading, uploadProgress } = useFileUpload(setMeshData, setPhase);

  return (
    <div className="upload-screen">
      <div className="upload-content">
        <h1 className="upload-title">3D Сегментация Печени</h1>
        
        {isUploading ? (
          <div className="upload-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="progress-text">
              Обработка... {uploadProgress}%
            </p>
          </div>
        ) : (
          <label className="upload-button">
            Загрузить медицинские данные
            <input
              type="file"
              style={{ display: 'none' }}
              onChange={(e) => {
                const file = e.target.files[0];
                if (file) handleFileUpload(file);
              }}
              accept=".nii,.nii.gz, .zip, .tar"
              disabled={isUploading}
            />
          </label>
        )}
        
        <p className="upload-hint">Поддерживаются файлы NIfTI (.nii, .nii.gz, .zip, .tar)</p>
      </div>
    </div>
  );
}