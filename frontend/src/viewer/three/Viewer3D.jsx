import { Canvas } from "@react-three/fiber";
import LiverMesh from "./LiverMesh";
import { OrbitControls, Grid } from "@react-three/drei";

export default function Viewer3D({ url }) {
  return (
    <div className="viewer-3d-wrapper">
      <Canvas camera={{ position: [100, 100, 100], fov: 60 }}>
        <color attach="background" args={["#1a1a1a"]} />
        <ambientLight intensity={0.5} />
        <directionalLight position={[50, 50, 50]} intensity={1} />
        <Grid args={[100, 100]} />
        <axesHelper args={[50]} />
        <LiverMesh url={url} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}