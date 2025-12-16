import { useState } from "react";
import { uploadSegmentation } from "./api/segmentation";
import { buildMesh } from "./api/mesh";

export default function App() {
  const [file, setFile] = useState(null);
  const [maskPath, setMaskPath] = useState(null);
  const [metrics, setMetrics] = useState(null);

  async function handleSegmentation() {
    if (!file) return alert("Файл не выбран");

    const result = await uploadSegmentation(file);
    setMaskPath(result.mask_path);
  }

  async function handleMesh() {
    if (!maskPath) return alert("Нет mask_path");

    const result = await buildMesh(maskPath);
    setMetrics(result.metrics);
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>Liver segmentation MVP</h1>

      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <br /><br />

      <button onClick={handleSegmentation}>
        1. Сегментировать
      </button>

      <br /><br />

      <button onClick={handleMesh}>
        2. Построить меш
      </button>

      {metrics && (
        <pre>{JSON.stringify(metrics, null, 2)}</pre>
      )}
    </div>
  );
}

