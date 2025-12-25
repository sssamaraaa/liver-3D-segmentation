import { useState } from "react";
import { buildMesh } from "../api/mesh";
import { runSegmentation } from "../api/segmentation";
import { useAppState } from "../app/appState";

export function useFileUpload() {
  const { setMeshData, setPhase, setProgress } = useAppState();
  const [isUploading, setIsUploading] = useState(false);

  async function handleFileUpload(file) {
    setIsUploading(true);
    setProgress(0);
    setPhase("processing");

    try {
      // ---------- 1. ЗАГРУЗКА ФАЙЛА С ПРОГРЕССОМ ----------
      const formData = new FormData();
      formData.append("file", file);

      const result = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "http://localhost:8000/segmentation/predict");

        xhr.upload.onprogress = e => {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 70); // 70% прогресса на upload
            setProgress(percent);
          }
        };

        xhr.onload = () => resolve(JSON.parse(xhr.responseText));
        xhr.onerror = reject;

        xhr.send(formData);
      });

      // ---------- 2. ОБРАБОТКА (сегментация + постпроцессинг) ----------
      // пока нет настоящего статуса с сервера, аккуратно докручиваем полоску
      for (let p = 70; p <= 90; p += 2) {
        await new Promise(r => setTimeout(r, 80)); // плавно 1 сек
        setProgress(p);
      }

      // ---------- 3. ПОСТРОЕНИЕ МЕША ----------
      const mesh = await buildMesh(result.mask_path);

      for (let p = 90; p <= 100; p += 2) {
        await new Promise(r => setTimeout(r, 50));
        setProgress(p);
      }

      setMeshData({ ...result, ...mesh });
      setPhase("done");

    } catch (err) {
      console.error("Upload error:", err);
      setPhase("idle");

    } finally {
      setIsUploading(false);
    }
  }

  return { handleFileUpload, isUploading };
}
