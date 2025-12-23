import { useDropzone } from "react-dropzone";

export default function DropZone({ onDrop, isUploading, onClose }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.nii', '.nii.gz']
    },
    multiple: false,
    disabled: isUploading
  });

  return (
    <div className="dropzone-overlay">
      <div 
        {...getRootProps()} 
        className={`dropzone ${isDragActive ? 'active' : ''}`}
      >
        <input {...getInputProps()} />
        
        {isUploading ? (
          <div className="dropzone-content">
            <div className="spinner"></div>
            <p>Обработка файла...</p>
          </div>
        ) : (
          <div className="dropzone-content">
            <p className="dropzone-title">
              {isDragActive ? 'Отпустите файл здесь' : 'Перетащите файл NIfTI сюда'}
            </p>
            <p className="dropzone-subtitle">или нажмите для выбора</p>
            <p className="dropzone-hint">Поддерживаются .nii и .nii.gz файлы</p>
          </div>
        )}
      </div>
      
      <button 
        className="dropzone-close"
        onClick={onClose}
        disabled={isUploading}
      >
        ×
      </button>
    </div>
  );
}