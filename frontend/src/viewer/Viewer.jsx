import { useAppState } from "../app/appState";
import Viewer3D from "./three/Viewer3D";
import SliceViewer from "./slices/SliceViewer";

export default function Viewer() {
  const { view, meshData } = useAppState();
  
  if (view === "3d") {
    const url = meshData?.files?.mesh_stl;
    const fullUrl = url ? `http://localhost:8000${url.startsWith('/')?url:'/'+url}` : null;

    return <Viewer3D url={fullUrl}/>;
  }

  return (
    <SliceViewer
      ctPath={meshData?.ct_path || meshData?.files?.ct}
      maskPath={meshData?.mask_path || meshData?.files?.mask}
      axis={view}
    />
  );
}
