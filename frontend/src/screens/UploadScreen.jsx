import AppLayout from "../app/AppLayout";
import { useFileUpload } from "../hooks/useFileUpload";
import UploadIcon from '../../assets/Insert.svg';

export default function UploadScreen() {
  const { handleFileUpload, isUploading } = useFileUpload();

  return (
    <AppLayout>
      <div className="upload-screen">
        <div className="upload-content">
          <div className="upload-area">
            <div className="upload-icon-circle">
              <img src={UploadIcon} alt="Upload icon" width="64" height="64"/>
            </div>

            <label className="upload-button">
              Загрузить данные
              <input
                type="file"
                style={{ display: 'none' }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                }}
                accept=".nii,.nii.gz,.zip,.tar"
                disabled={isUploading}
              />
            </label>

            <p className="upload-formats">
              Поддерживаемые форматы: .nii, .nii.gz, .zip, .tar
            </p>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
