import { useState } from "react";
import { runSegmentation } from "./api/segmentation";
import { buildMesh } from "./api/mesh";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);

  async function handleFile(file) {
    try {
      setLoading(true);

      const seg = await runSegmentation(file);
      const mesh = await buildMesh(seg.mask_path);

      setMetrics(mesh.metrics);
    } catch (e) {
      console.error(e);
      alert("Ошибка обработки");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen w-screen bg-zinc-900 text-zinc-100 flex items-center justify-center">
      {!metrics && !loading && (
        <label className="bg-emerald-600 px-4 py-2 rounded cursor-pointer">
          Загрузить файл
          <input
            type="file"
            accept=".nii,.nii.gz"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </label>
      )}

      {loading && <div>Обработка данных…</div>}

      {metrics && (
        <div className="space-y-2">
          <div>Объём: {metrics.volume_ml} мл</div>
          <div>Площадь: {metrics.surface_mm2} мм²</div>
        </div>
      )}
    </div>
  );
}
