import { useAppState } from "../app/appState";
import Viewer3D from "./three/Viewer3D";
import SliceViewer from "./slices/SliceViewer";

export default function Viewer() {
  const { view, meshData } = useAppState();
  
  if (view === "3d") {
    const meshUrl = meshData?.files?.mesh_stl;
    const correctedUrl = meshUrl ? `http://localhost:8000${meshUrl.startsWith('/') ? meshUrl : `/${meshUrl}`}` : null;
    
    return (
      <div style={{
        width: '100%',
        height: '100%',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0
      }}>
        <Viewer3D url={correctedUrl} />
      </div>
    );
  }
  
    const ctPath = meshData?.ct_path || meshData?.files?.ct;
    const maskPath = meshData?.mask_path || meshData?.files?.mask;

  return (
    <SliceViewer
      ctPath={ctPath}
      maskPath={maskPath}
      axis={view}
    />
  );
}