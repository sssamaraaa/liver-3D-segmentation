import { useState } from "react";
import { runSegmentation } from "../api/segmentation";
import { buildMesh } from "../api/mesh";

export function useFileUpload(setMeshData, setPhase) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileUpload = async (file) => {
    try {
      setIsUploading(true);
      setUploadProgress(0);
      setPhase("processing");

      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      const seg = await runSegmentation(file);
      setUploadProgress(50);

      const mesh = await buildMesh(seg.mask_path);
      setUploadProgress(90);

      const correctedMesh = {
        ...mesh,
        ct_path: seg.ct_path.replace(/\\/g, "/"),
        mask_path: seg.mask_path.replace(/\\/g, "/"),
        files: {
          ...mesh.files,
          mesh_stl: '/' + mesh.files.mesh_stl.replace(/\\/g, '/'),
          mesh_ply: '/' + mesh.files.mesh_ply.replace(/\\/g, '/'),
          ct: seg.ct_path.replace(/\\/g, "/"),
          mask: seg.mask_path.replace(/\\/g, "/")
        }
      };

      clearInterval(progressInterval);
      setUploadProgress(100);

      setMeshData(correctedMesh);
      setPhase("ready");
      
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
      }, 1000);

      return correctedMesh;
    } catch (e) {
      console.error("Error:", e);
      alert("Ошибка обработки файла");
      setPhase("idle");
      setIsUploading(false);
      setUploadProgress(0);
      throw e;
    }
  };

  return {
    handleFileUpload,
    isUploading,
    uploadProgress
  };
}