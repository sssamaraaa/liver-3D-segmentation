import { runSegmentation } from "../api/segmentation";
import { buildMesh } from "../api/mesh";
import { useAppState } from "../app/appState";

export default function UploadScreen() {
  const { setPhase, setMeshData } = useAppState();

  async function handleFile(file) {
    try {
      setPhase("processing");
      const seg = await runSegmentation(file);
      const mesh = await buildMesh(seg.mask_path);
      
      const correctedMesh = {
        ...mesh,
        files: {
          ...mesh.files,
          mesh_stl: '/' + mesh.files.mesh_stl.replace(/\\/g, '/'),
          mesh_ply: '/' + mesh.files.mesh_ply.replace(/\\/g, '/'),
          mask: mesh.files.mask
        }
      };
      
      setMeshData(correctedMesh);
      setPhase("ready");
    } catch (e) {
      console.error("Error:", e);
      alert("Ошибка обработки");
      setPhase("idle");
    }
  }

  return (
    <div className="upload-screen">
      <div className="upload-content">
        <h1 className="upload-title">3D Сегментация Печени</h1>
        <label className="upload-button">
          Загрузить медицинские данные
          <input
            type="file"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
            accept=".nii,.nii.gz"
          />
        </label>
        <p className="upload-hint">Поддерживаются файлы NIfTI (.nii, .nii.gz)</p>
      </div>
    </div>
  );
}