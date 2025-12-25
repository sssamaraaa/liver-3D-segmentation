import { useState } from "react";
import { buildMesh } from "../api/mesh";
import { useAppState } from "../app/appState";

export function useFileUpload() {
  const { setMeshData, setPhase, setProgress } = useAppState();
  const [isUploading, setIsUploading] = useState(false);

  async function handleFileUpload(file) {
    setIsUploading(true);
    setProgress(0);
    setPhase("processing");

    try {

      const formData = new FormData();
      formData.append("file", file);

      const result = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", "http://localhost:8000/segmentation/predict");

        xhr.upload.onprogress = e => {
          if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 70); 
            setProgress(percent);
          }
        };

        xhr.onload = () => resolve(JSON.parse(xhr.responseText));
        xhr.onerror = reject;

        xhr.send(formData);
      });

      for (let p = 70; p <= 90; p += 2) {
        await new Promise(r => setTimeout(r, 80)); 
        setProgress(p);
      }

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
